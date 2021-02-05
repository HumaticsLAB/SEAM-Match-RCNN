from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss, maskrcnn_inference
from torchvision.ops import boxes as box_ops

from .match_head import MatchPredictor, MatchLoss, TemporalAggregationNLB as TemporalAggregation

model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}


class TemporalRoIHeads(torch.nn.Module):

    def __init__(self, orh, n_frames):
        # orh: old_roi_heads
        super(TemporalRoIHeads, self).__init__()
        self.n_frames = n_frames

        self.box_roi_pool = orh.box_roi_pool
        self.box_head = orh.box_head
        self.box_predictor = orh.box_predictor

        self.mask_roi_pool = orh.mask_roi_pool
        self.mask_head = orh.mask_head
        self.mask_predictor = orh.mask_predictor

        self.match_predictor = MatchPredictor()
        self.match_loss = MatchLoss()

        self.temporal_aggregator = TemporalAggregation()

        self.keypoint_roi_pool = orh.keypoint_head
        self.keypoint_head = orh.keypoint_head
        self.keypoint_predictor = orh.keypoint_predictor

        self.score_thresh = orh.score_thresh
        self.nms_thresh = orh.nms_thresh
        self.detections_per_img = orh.detections_per_img

        self.proposal_matcher = orh.proposal_matcher
        self.fg_bg_sampler = orh.fg_bg_sampler
        self.box_coder = orh.box_coder

        self.box_similarity = box_ops.box_iou

    @property
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    @property
    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    @property
    def has_match(self):
        if self.match_predictor is None:
            return False
        if self.match_loss is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("labels" in t for t in targets)
        if self.has_mask:
            assert all("masks" in t for t in targets)

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                if boxes[i].numel() > 0:
                    result.append(
                        dict(
                            boxes=boxes[i],
                            labels=labels[i],
                            scores=scores[i],
                        )
                    )
                else:
                    result.append(
                        dict(
                            boxes=torch.tensor([0.0, 0.0, image_shapes[i][1], image_shapes[i][0]]).to(boxes[i].device).unsqueeze(0),
                            labels=torch.tensor([0]).to(boxes[i].device),
                            scores=torch.tensor([0.1]).to(boxes[i].device),
                        )
                    )

        if self.has_mask:
            if targets is not None:
                assert(len(targets) == len(result))
                # result.extend([{k:v for k, v in x.items() if k in ["boxes", "labels"]} for x in targets])
                for i, r in enumerate(result):
                    r["boxes"] = torch.cat([targets[i]["boxes"], r["boxes"]])
                    r["labels"] = torch.cat([targets[i]["labels"], r["labels"]])
                    r["scores"] = torch.cat([torch.ones((targets[i]["labels"].numel(),)).to(r["scores"].device), r["scores"]])
                    # if "scores" not in r:
                    #     r["scores"] = torch.ones((r["labels"].numel(),))
            mask_proposals = [p["boxes"] for p in result]

            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            mask_roi_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_roi_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        if self.has_match:
            if self.training:
                types = []
                s_imgs = []
                i = 0
                for p, s in zip(mask_proposals, targets):
                    types = types + ([1] * len(p) if s['sources'][0] == 1 else [0] * len(p))
                    s_imgs = s_imgs + ([i] * len(p))
                    i += 1
                types = torch.IntTensor(types)
                final_features, match_logits = self.match_predictor(mask_roi_features, types)

                gt_pairs = [t["pair_ids"] for t in targets]
                gt_styles = [t["styles"] for t in targets]

                loss_match = self.match_loss(match_logits, mask_proposals, gt_pairs, gt_styles, types, pos_matched_idxs)
                # print(loss_match)
                loss_match = dict(loss_match=loss_match)


            else:
                loss_match = {}
                s_imgs = []
                for i, p in enumerate(mask_proposals):
                    if i == 0:
                        types = [0] * len(p)
                    else:
                        types = types + [1] * len(p)
                    s_imgs = s_imgs + ([i] * len(p))

                types = torch.IntTensor(types)
                if mask_roi_features.shape[0] > 0:
                    final_features, match_logits = self.match_predictor(mask_roi_features, types)
                    for i, r in zip(range(len(mask_proposals)), result):
                        r['match_features'] = final_features[torch.IntTensor(s_imgs) == i, ...]
                        r['w'] = self.match_predictor.last.weight
                        r['b'] = self.match_predictor.last.bias
                        r['roi_features'] = mask_roi_features[torch.IntTensor(s_imgs) == i]

            losses.update(loss_match)

        return result, losses



class VideoMatchRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes, n_frames, **kwargs):
        super(VideoMatchRCNN, self).__init__(backbone, num_classes, **kwargs)
        self.roi_heads = TemporalRoIHeads(self.roi_heads, n_frames)

    def load_saved_matchrcnn(self, sd):
        self.load_state_dict(sd, strict=False)
        self.roi_heads.tracking_predictor\
            .load_state_dict(deepcopy(self.roi_heads.match_predictor.state_dict()))
        self.roi_heads.temporal_aggregator\
            .load_state_dict(deepcopy(self.roi_heads.match_predictor.state_dict()), strict=False)


def videomatchrcnn_resnet50_fpn(pretrained=False, progress=True,
                                num_classes=91, pretrained_backbone=True,
                                n_frames=3, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = VideoMatchRCNN(backbone, num_classes, n_frames, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
