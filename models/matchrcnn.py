import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torch.hub import load_state_dict_from_url
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_loss, maskrcnn_inference
from .match_head import MatchPredictor, MatchLossPreTrained, filter_proposals


from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

params = {
    'rpn_anchor_generator': AnchorGenerator((32, 64, 128, 256, 512), (0.5, 1.0, 2.0)),
    'rpn_pre_nms_top_n_train': 2000,
    'rpn_pre_nms_top_n_test': 1000,
    'rpn_post_nms_top_n_test': 4000,
    'rpn_post_nms_top_n_train': 8000,

    'box_roi_pool': MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2),
    'mask_roi_pool': MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2),
    # 'image_mean': [0.60781138, 0.57332054, 0.55193729],
    # 'image_std': [0.06657078, 0.06587644, 0.06175072]
}


model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}


class NewRoIHeads(torch.nn.Module):
    def __init__(self, orh):
        # orh: old_roi_heads
        super(NewRoIHeads, self).__init__()

        self.box_roi_pool = orh.box_roi_pool
        self.box_head = orh.box_head
        self.box_predictor = orh.box_predictor

        self.mask_roi_pool = orh.mask_roi_pool
        self.mask_head = orh.mask_head
        self.mask_predictor = orh.mask_predictor

        self.match_predictor = MatchPredictor()
        self.match_loss = MatchLossPreTrained()

        # self.keypoint_roi_pool = MultiScaleRoIAlign(
        #     featmap_names=[0, 1, 2, 3],
        #     output_size=14,
        #     sampling_ratio=2)
        #
        # keypoint_layers = tuple(512 for _ in range(8))
        # self.keypoint_head = KeypointRCNNHeads(256, keypoint_layers)
        # self.keypoint_predictor = KeypointRCNNPredictor(512, num_keypoints=294)
        #
        # #
        self.keypoint_roi_pool = None
        self.keypoint_head = None
        self.keypoint_predictor = None

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

    def keypoints_to_heatmap(self, keypoints, rois, heatmap_size):
        offset_x = rois[:, 0]
        offset_y = rois[:, 1]
        scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
        scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

        offset_x = offset_x[:, None]
        offset_y = offset_y[:, None]
        scale_x = scale_x[:, None]
        scale_y = scale_y[:, None]

        x = keypoints[..., 0]
        y = keypoints[..., 1]

        x_boundary_inds = x == rois[:, 2][:, None]
        y_boundary_inds = y == rois[:, 3][:, None]

        x = (x - offset_x) * scale_x
        x = x.floor().long()
        y = (y - offset_y) * scale_y
        y = y.floor().long()

        x[x_boundary_inds] = heatmap_size - 1
        y[y_boundary_inds] = heatmap_size - 1

        valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
        vis = keypoints[..., 2] > 0
        valid = (valid_loc & vis).long()

        lin_ind = y * heatmap_size + x
        heatmaps = lin_ind * valid

        return heatmaps, valid

    def heatmaps_to_keypoints(self, maps, rois):
        """Extract predicted keypoint locations from heatmaps. Output has shape
        (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
        for each keypoint.
        """
        # This function converts a discrete image coordinate in a HEATMAP_SIZE x
        # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
        # consistency with keypoints_to_heatmap_labels by using the conversion from
        # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
        # continuous coordinate.
        offset_x = rois[:, 0]
        offset_y = rois[:, 1]

        widths = rois[:, 2] - rois[:, 0]
        heights = rois[:, 3] - rois[:, 1]
        widths = widths.clamp(min=1)
        heights = heights.clamp(min=1)
        widths_ceil = widths.ceil()
        heights_ceil = heights.ceil()

        num_keypoints = maps.shape[1]
        xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
        end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
        for i in range(len(rois)):
            roi_map_width = int(widths_ceil[i].item())
            roi_map_height = int(heights_ceil[i].item())
            width_correction = widths[i] / roi_map_width
            height_correction = heights[i] / roi_map_height
            roi_map = torch.nn.functional.interpolate(
                maps[i][None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[0]
            # roi_map_probs = scores_to_probs(roi_map.copy())
            w = roi_map.shape[2]
            pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)
            x_int = pos % w
            y_int = (pos - x_int) // w
            # assert (roi_map_probs[k, y_int, x_int] ==
            #         roi_map_probs[k, :, :].max())
            x = (x_int.float() + 0.5) * width_correction
            y = (y_int.float() + 0.5) * height_correction
            xy_preds[i, 0, :] = x + offset_x[i]
            xy_preds[i, 1, :] = y + offset_y[i]
            xy_preds[i, 2, :] = 1
            end_scores[i, :] = roi_map[torch.arange(num_keypoints), y_int, x_int]

        return xy_preds.permute(0, 2, 1), end_scores

    def keypointrcnn_loss(self, keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs):
        N, K, H, W = keypoint_logits.shape
        assert H == W
        discretization_size = H
        heatmaps = []
        valid = []

        indx = [x for x in gt_keypoints]

        for proposals_per_image, gt_kp_in_image, midx in zip(proposals, gt_keypoints, keypoint_matched_idxs):
            kp = gt_kp_in_image[midx]
            heatmaps_per_image, valid_per_image = self.keypoints_to_heatmap(
                kp, proposals_per_image, discretization_size
            )
            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))

        keypoint_targets = torch.cat(heatmaps, dim=0)
        valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

        # torch.mean (in binary_cross_entropy_with_logits) does'nt
        # accept empty tensors, so handle it sepaartely
        if keypoint_targets.numel() == 0 or len(valid) == 0:
            return keypoint_logits.sum() * 0

        keypoint_logits = keypoint_logits.view(N * K, H * W)

        keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
        return keypoint_loss

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

        box_features_roi = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features_roi)
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
                            boxes=torch.tensor([0.0, 0.0, image_shapes[i][1], image_shapes[i][0]]).to(
                                boxes[i].device).unsqueeze(0),
                            labels=torch.tensor([0]).to(boxes[i].device),
                            scores=torch.tensor([1.0]).to(boxes[i].device),
                        )
                    )

        if self.has_mask:
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
            match_proposals = [p["boxes"] for p in result]
            if self.training:
                gt_proposals = [t["boxes"] for t in targets]
                num_images = len(proposals)
                match_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    match_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

                match_roi_features = self.mask_roi_pool(features, match_proposals, image_shapes)
                match_proposals, mask_roi_features, matched_idxs_match = filter_proposals(match_proposals,
                                                                                          match_roi_features,
                                                                                          gt_proposals,
                                                                                          pos_matched_idxs)
                types = []
                s_imgs = []
                i = 0
                for p, s in zip(match_proposals, targets):
                    types = types + ([1] * len(p) if s['sources'][0] == 1 else [0] * len(p))
                    s_imgs = s_imgs + ([i] * len(p))
                    i += 1
                types = torch.IntTensor(types)
                # match_roi_features = self.mask_roi_pool(features, match_proposals, image_shapes)
                final_features, match_logits = self.match_predictor(mask_roi_features, types)

                gt_pairs = [t["pair_ids"] for t in targets]
                gt_styles = [t["styles"] for t in targets]

                loss_match = self.match_loss(match_logits, match_proposals, gt_proposals, gt_pairs, gt_styles, types,
                                             pos_matched_idxs)

                loss_match = dict(loss_match=loss_match)


            else:
                loss_match = {}

                s_imgs = []
                for i, p in enumerate(match_proposals):
                    if i == 0:
                        types = [0] * len(p)
                    else:
                        types = types + [1] * len(p)
                    s_imgs = s_imgs + ([i] * len(p))

                types = torch.IntTensor(types)
                match_roi_features = self.mask_roi_pool(features, match_proposals, image_shapes)
                final_features, match_logits = self.match_predictor(match_roi_features, types)
                for i, r in zip(range(len(match_proposals)), result):
                    r['match_features'] = final_features[torch.IntTensor(s_imgs) == i, ...]
                    r['w'] = self.match_predictor.last.weight
                    r['b'] = self.match_predictor.last.bias

            losses.update(loss_match)

        return result, losses


class MatchRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes, **kwargs):
        super(MatchRCNN, self).__init__(backbone, num_classes, **kwargs)
        self.roi_heads = NewRoIHeads(self.roi_heads)


def matchrcnn_resnet50_fpn(pretrained=False, progress=True,
                           num_classes=91, pretrained_backbone=True, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = MatchRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
