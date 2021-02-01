import torch
from torch import nn
from torch.nn import functional as F
from pycocotools import mask as maskUtils
from .nlb import NONLocalBlock1D


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


class MatchPredictor(nn.Module):
    def __init__(self):
        super(MatchPredictor, self).__init__()
        self.conv_seq = nn.Sequential(nn.Conv2d(256, 256, 3),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, 3),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 1024, 3),
                                      nn.ReLU(),
                                      )
        self.pool = nn.Sequential(nn.AvgPool2d((6, 6)),
                                  nn.ReLU(), )
        self.linear = nn.Sequential(nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256), )

        self.last = nn.Linear(256, 2)

    def forward(self, x, types):
        x1 = self.conv_seq(x)
        x2 = self.pool(x1)
        x3 = self.linear(x2.view(x2.size(0), -1))
        # TODO:  after linear there should be Substraction between different pairs, and element wise square...
        # then, we can apply the final linear transformation (last) and the softmax function (output)
        x3_1 = x3[types == 0].unsqueeze(1)
        x3_2 = x3[types == 1].unsqueeze(0)

        x4 = (x3_1 - x3_2) ** 2
        x5 = self.last(x4)
        # return x3, F.softmax(x5, dim=-1)
        return x3, x5


class TemporalAggregationNLB(MatchPredictor):

    def __init__(self, d_model=256):
        super(TemporalAggregationNLB, self).__init__()
        # same parameters and same forward as standard MatchPredictor
        # except for temporal aggregation
        self.n_frames = -1
        self.attention_scorer = nn.Linear(d_model, 1)
        self.newnlb = NONLocalBlock1D(in_channels=d_model, sub_sample=False, bn_layer=False)
        self.nlb = True

    def forward(self, x, types, ids, x3_1_seq=None, x3_1_mask=None, x3_2=None, getatt=False):
        # x should be (K*(n_frames + 1))x256x14x14 where the one is shop and the n_frames are frames
        if x3_1_seq is None:
            x1 = self.conv_seq(x)
            x2 = self.pool(x1)
            x3 = self.linear(x2.view(x2.size(0), -1))
            x3_1 = x3[types == 0]  # should be (K*n_frames)x256
            x3_1_ids = ids[types == 0]
            if x3_1_ids.numel() > 0:
                maxlen = (x3_1_ids == x3_1_ids.mode()[0]).sum()
                n_seqs = x3_1_ids.unique().numel()
                # first token is a dummy where the output is going to be
                x3_1_seq = torch.zeros((1 + maxlen, n_seqs, 256), device=x3_1.device, dtype=x3_1.dtype, requires_grad=False)
                # True values are to be masked
                # https://github.com/pytorch/pytorch/blob/5f25e98fc758ab2f32791364d855be8ff9cb36e7/torch/nn/modules/transformer.py#L66
                x3_1_mask = torch.zeros((n_seqs, 1 + maxlen), device=x3_1.device, dtype=torch.bool)
                x3_1_list = []
                for i, idd in enumerate(x3_1_ids.unique()):
                    tmp_n = (x3_1_ids == idd).sum().item()
                    x3_1_seq[1:tmp_n + 1, i] = x3_1[x3_1_ids == idd]
                    x3_1_mask[i, tmp_n + 1:] = 1
                    x3_1_list.append(x3_1[x3_1_ids == idd])


                if self.nlb:
                    x3_1_list = [self.newnlb(x.transpose(0, 1).unsqueeze(0))[0].transpose(0, 1)
                                 if x.shape[0] > 1 else x
                                 for x in x3_1_list]

                x3_1b = [(F.softmax(self.attention_scorer(x), 0) * x3_1_list[i]).sum(0).unsqueeze(0)
                         for i, x in enumerate(x3_1_list)]
                x3_1b = torch.cat(x3_1b, 0)

                if getatt:
                    attention_scores = [F.softmax(self.attention_scorer(x), 0) for x in x3_1_list]

                x3_1c = x3_1b.unsqueeze(1)
            else:
                x3_1b = None
                x3_1c = None

            x3_2 = x3[types == 1]
            x3_2b = x3_2.unsqueeze(0)
        else:

            # build list
            x3_1_inds = [(x3_1_mask[i]).nonzero()[0].item() if (x3_1_mask[i]).any()
                         else x3_1_mask[i].numel()
                         for i in range(x3_1_seq.shape[1])]
            x3_1_list = [x3_1_seq[1:x3_1_inds[i], i] for i in range(x3_1_seq.shape[1])]




            if self.nlb:
                x3_1_list = [self.newnlb(x.transpose(0, 1).unsqueeze(0))[0].transpose(0, 1)
                                 if x.shape[0] > 1 else x
                                 for x in x3_1_list]

            x3_1b = [(F.softmax(self.attention_scorer(x), 0) * x3_1_list[i]).sum(0).unsqueeze(0)
                         for i, x in enumerate(x3_1_list)]
            x3_1b = torch.cat(x3_1b, 0)

            if getatt:
                attention_scores = [F.softmax(self.attention_scorer(x), 0) for x in x3_1_list]

            x3_1c = x3_1b.unsqueeze(1)
            x3_2b = x3_2.unsqueeze(0)
            x3_1_ids = torch.zeros((1, 2))  # just to have numel > 0

        if x3_1_ids.numel() > 0:
            x4 = (x3_1c - x3_2b) ** 2
            x5 = self.last(x4)
        else:
            x5 = None

        if getatt:
            return x3_1b, x3_2, x5, x3_1_seq, x3_1_mask, x3_1_ids, attention_scores

        return x3_1b, x3_2, x5, x3_1_seq, x3_1_mask, x3_1_ids


class MatchLoss(object):
    def __init__(self):
        super(MatchLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, logits, proposals, gt_pairs, gt_styles, types, matched_idxs):

        target_pairs = [l[idxs] for l, idxs in zip(gt_pairs, matched_idxs)]
        target_styles = [l[idxs] for l, idxs in zip(gt_styles, matched_idxs)]
        # target_pairs, target_styles = self._prepare_target(proposals, targets)

        target_pairs_user = torch.cat(target_pairs)[types == 0]
        target_styles_user = torch.cat(target_styles)[types == 0]

        target_pairs_shop = torch.cat(target_pairs)[types == 1]
        target_styles_shop = torch.cat(target_styles)[types == 1]

        gts = torch.zeros(len(target_pairs_user), len(target_pairs_shop), dtype=torch.int64).to(logits.device)
        for i in range(len(target_pairs_user)):
            for j in range(len(target_pairs_shop)):
                tpu = target_pairs_user[i]
                tps = target_pairs_shop[j]
                tsu = target_styles_user[i]
                tss = target_styles_shop[j]
                if tps == tpu and tsu == tss:
                    gts[i, j] = 1
                else:
                    gts[i, j] = 0

        gts = gts.view(-1)
        logits = logits.view(-1, 2)

        loss = self.criterion(logits, gts)
        if loss > 1.0:
            loss = loss / 2.0
        return loss


class MatchLossWeak(object):

    def __init__(self, device, match_threshold=-10.0):
        super(MatchLossWeak, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
        self.match_threshold = match_threshold

    def __call__(self, logits, types, prod_ids, img_ids):
        img_ids = torch.tensor(img_ids)
        prod_ids = torch.tensor(prod_ids)
        gts = torch.zeros(logits.shape[0], logits.shape[1], dtype=torch.int64).to(logits.device)
        street_inds = (types == 0).nonzero().view(-1)
        shop_inds = (types == 1).nonzero().view(-1)
        # associate to each detection the corresponding index in the logits
        reverse_street_shop_inds = torch.zeros_like(types, dtype=torch.int64)
        reverse_street_shop_inds[street_inds] = torch.arange(street_inds.shape[0])
        reverse_street_shop_inds[shop_inds] = torch.arange(shop_inds.shape[0])
        for ii in torch.unique(img_ids):
            # get data on detections of this image
            tmp_type = int(types[img_ids == ii][0])
            if tmp_type == 1:
                continue
            # da qui solo se ho uno street
            tmp_prod_id = int(prod_ids[img_ids == ii][0])
            tmp_inds = (img_ids == ii).nonzero().view(-1)
            # cerco lo shop corrispondente
            shop_ind = ((prod_ids == tmp_prod_id) & (types == 1)).nonzero().view(-1)
            tmp_logits = logits[reverse_street_shop_inds[tmp_inds], reverse_street_shop_inds[shop_ind], 1].view(-1)
            max_score, max_score_ind = tmp_logits.max(), tmp_inds[tmp_logits.argmax()]

            if max_score > self.match_threshold:
                gts[reverse_street_shop_inds[max_score_ind], reverse_street_shop_inds[shop_ind]] = 1

        gts = gts.view(-1)
        logits = logits.view(-1, 2)
        loss = self.criterion(logits, gts)
        return loss

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


class NEWBalancedAggregationMatchLossWeak(object):

    def __init__(self, device, temporal_aggregator, match_threshold=-10.0):
        super(NEWBalancedAggregationMatchLossWeak, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.3]).to(device))
        self.match_threshold = match_threshold
        self.temporal_aggregator = temporal_aggregator

    def __call__(self, match_logits, types, prod_ids, img_ids, roi_features):
        '''
        Calcola la loss di matching calcolando il temporal aggregator insieme. Serve fare tutto insieme perchè
        il temporal aggregator necessita di dati in input in un'ordine particolare
        :param match_logits: gli score di matching street2shop
        :param types: i tipi street (0) o shop (1)
        :param prod_ids: id dei prodotti
        :param img_ids: id delle immagini nel batch
        :return: loss
        '''
        img_ids = torch.tensor(img_ids)
        prod_ids = torch.tensor(prod_ids)
        street_inds = (types == 0).nonzero().view(-1)
        shop_inds = (types == 1).nonzero().view(-1)
        # associate to each detection the corresponding index in the logits
        reverse_street_shop_inds = torch.zeros_like(types, dtype=torch.int64)
        reverse_street_shop_inds[street_inds] = torch.arange(street_inds.shape[0])
        reverse_street_shop_inds[shop_inds] = torch.arange(shop_inds.shape[0])
        aggregation_candidates = []

        for pi in torch.unique(prod_ids):
            tmp_prod_inds = (prod_ids == pi).nonzero().view(-1)
            for iiind, ii in enumerate(torch.unique(img_ids[tmp_prod_inds])):
                tmp_type = int(types[img_ids == ii][0])
                if tmp_type == 1:

                    continue

                tmp_prod_id = pi
                tmp_inds = (img_ids == ii).nonzero().view(-1)
                shop_ind = ((prod_ids == tmp_prod_id) & (types == 1)).nonzero().view(-1)
                tmp_logits = match_logits[reverse_street_shop_inds[tmp_inds], reverse_street_shop_inds[shop_ind], 1].view(-1)
                # max_score_ind is the index within the boxes in this image id
                max_score, max_score_ind = tmp_logits.max(), tmp_logits.argmax()
                if max_score > self.match_threshold:
                    # save the corresponding tracking ind
                    aggregation_candidates.append(tmp_inds[max_score_ind])
        if len(aggregation_candidates) == 0:
            # doesn't find enough aggregation candidates
            return torch.tensor(0, dtype=torch.float32).to(match_logits.device)
        # tracking_candidates will contain the "best" box for each image
        aggregation_candidates = torch.tensor(aggregation_candidates)
        # CLEAN TRACKING LOGITS

        valid_prods = []
        street_feature_inds = []
        gt_flag = []
        seq_ids = []
        seq_count = 0
        # build aggregation combinations
        for pi in torch.unique(prod_ids[aggregation_candidates]):
            tmp_cands = aggregation_candidates[prod_ids[aggregation_candidates] == pi]
            if tmp_cands.numel() < self.temporal_aggregator.n_frames:
                continue
            valid_prods.append(pi)

            tmp_combs = tmp_cands
            street_feature_inds.append(tmp_combs)
            gt_flag.append(torch.tensor([1] * tmp_combs.numel()))
            seq_ids.append(torch.tensor([seq_count] * tmp_combs.numel()))
            seq_count += 1



        if len(valid_prods) == 0:
            # doesn't find enough valid frames
            return torch.tensor(0, dtype=torch.float32).to(roi_features.device)
        # products for which we have at least n_frames frames (we can compute aggregation matching for them)
        # we take the shop images for them
        valid_prods = torch.tensor(valid_prods)

        shop_feature_inds = []
        for pi in valid_prods:
            shop_ind = ((prod_ids == pi) & (types == 1)).nonzero().view(-1)
            shop_feature_inds.append(shop_ind)
        shop_feature_inds = torch.tensor(shop_feature_inds)
        # used to index features in groups of n_frames, by repeating them
        street_feature_inds = torch.cat(street_feature_inds)
        gt_flag = torch.cat(gt_flag)
        seq_ids = torch.cat(seq_ids)

        feature_inds = torch.cat([street_feature_inds, shop_feature_inds])
        seq_ids = torch.cat([seq_ids]
                            + [torch.tensor(i + seq_count).view(-1) for i in range(shop_feature_inds.numel())])
        new_roi_features = roi_features[feature_inds]
        new_types = types[feature_inds]

        _, _, aggregator_logits, _, _, _ = self.temporal_aggregator(new_roi_features, new_types, seq_ids)

        # gts has a row for every subset of frames and a columns for every valid shop product

        gts = torch.zeros(seq_ids[:street_feature_inds.numel()].unique().numel()
                          , shop_feature_inds.numel(), dtype=torch.int64).to(aggregator_logits.device)
        # for every subset of frames
        for i, seq_id in enumerate(seq_ids[:street_feature_inds.numel()].unique()):
            # inds of this sequence
            street_inds = (seq_ids == seq_id).nonzero().view(-1)
            seq_inds = street_feature_inds[street_inds]
            # prod_id of this sequence
            tmp_prod_id = prod_ids[seq_inds][0]
            # find the column for it
            j = (tmp_prod_id == valid_prods).nonzero().view(-1)
            # set gt to true
            gts[i, j] = gt_flag[street_inds[0]]

        gts = gts.view(-1)
        logits = aggregator_logits.view(-1, 2)
        loss = self.criterion(logits, gts)
        return loss


class MatchLossDF2(object):

    def __init__(self, device):
        super(MatchLossDF2, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))

    def __call__(self, logits, types, raw_gt):
        street_inds = (types == 0).nonzero().view(-1)
        shop_inds = (types == 1).nonzero().view(-1)
        raw_gt = torch.tensor(raw_gt)
        shop_prods = raw_gt[shop_inds]
        street_prods = raw_gt[street_inds]
        gts = shop_prods.unsqueeze(0) == street_prods.unsqueeze(1)
        gts = gts.view(-1).to(logits.device).to(torch.int64)
        logits = logits.view(-1, 2)
        loss = self.criterion(logits, gts)
        return loss


class AggregationMatchLossDF2(object):

    def __init__(self, device, temporal_aggregator):
        super(AggregationMatchLossDF2, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.3]).to(device))
        self.temporal_aggregator = temporal_aggregator

    def __call__(self, types, roi_features, raw_gt):
        '''
        Calcola la loss di matching calcolando il temporal aggregator insieme. Serve fare tutto insieme perchè
        il temporal aggregator necessita di dati in input in un'ordine particolare
        :param match_logits: gli score di matching street2shop
        :param types: i tipi street (0) o shop (1)
        :param prod_ids: id dei prodotti
        :param img_ids: id delle immagini nel batch
        :return: loss
        '''
        street_inds = (types == 0).nonzero().view(-1)
        shop_inds = (types == 1).nonzero().view(-1)
        raw_gt = torch.tensor(raw_gt)
        unprods = raw_gt.unique()
        unprods = unprods[unprods > 0]


        valid_prods = []
        street_feature_inds = []
        seq_ids = []

        seq_count = 0
        # build aggregation combinations
        for pi in unprods:
            tmp_combs = street_inds[raw_gt[street_inds] == pi]
            if tmp_combs.numel() < 3: #self.temporal_aggregator.n_frames:
                continue
            valid_prods.append(pi)
            street_feature_inds.append(tmp_combs)
            seq_ids.append(torch.tensor([seq_count] * tmp_combs.numel()))
            seq_count += 1
        valid_prods = torch.tensor(valid_prods)
        street_feature_inds = torch.cat(street_feature_inds)
        try:
            seq_ids = torch.cat(seq_ids)
        except:
            print(seq_ids)
            quit()

        shop_feature_inds = shop_inds
        shop_feature_inds = torch.tensor(shop_feature_inds)

        feature_inds = torch.cat([street_feature_inds, shop_feature_inds])
        seq_ids = torch.cat([seq_ids]
                            + [torch.tensor(i + seq_count).view(-1) for i in range(shop_feature_inds.numel())])
        new_roi_features = roi_features[feature_inds]
        new_types = types[feature_inds]

        _, _, aggregator_logits, _, _, _ = self.temporal_aggregator(new_roi_features, new_types, seq_ids)

        shop_prods = raw_gt[shop_inds]
        street_prods = valid_prods
        gts = shop_prods.unsqueeze(0) == street_prods.unsqueeze(1)

        gts = gts.view(-1).to(aggregator_logits.device).to(torch.int64)
        logits = aggregator_logits.view(-1, 2)
        loss = self.criterion(logits, gts)
        return loss


def filter_proposals(proposals, mask_roi_features, gt_proposals, matched_idxs):
    match_imgs_mask = []
    new_mask_roi_features = []
    for i, pr_prop in enumerate(proposals):
        g_prop = gt_proposals[i]
        match_idxs = matched_idxs[i]
        match_imgs_mask = match_imgs_mask + ([i] * match_idxs.size(0))

        n_valid = g_prop.size(0)
        ious = torch.FloatTensor(
            maskUtils.iou(pr_prop.detach().cpu().numpy(), g_prop.detach().cpu().numpy(),
                          [0] * n_valid)).squeeze()
        if len(pr_prop) > 1:
            topKidxs = torch.argsort(ious, descending=True, dim=0)[
                       :torch.min(torch.tensor([(8 // n_valid), len(pr_prop)]))].view(-1)
            proposals[i] = pr_prop[topKidxs, :]
            matched_idxs[i] = match_idxs[topKidxs]
            new_mask_roi_features.append(mask_roi_features[torch.where(torch.IntTensor(match_imgs_mask) == i)[0], ...])
            new_mask_roi_features[i] = new_mask_roi_features[i][topKidxs, ...]
        else:
            new_mask_roi_features.append(mask_roi_features[torch.where(torch.IntTensor(match_imgs_mask) == i)[0], ...])

    return proposals, torch.cat(new_mask_roi_features), matched_idxs


class MatchLossPreTrained(object):
    def __init__(self):
        super(MatchLossPreTrained, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, logits, proposals, gt_proposals, gt_pairs, gt_styles, types, matched_idxs):

        target_pairs = [l[idxs] for l, idxs in zip(gt_pairs, matched_idxs)]
        target_styles = [l[idxs] for l, idxs in zip(gt_styles, matched_idxs)]
        # target_pairs, target_styles = self._prepare_target(proposals, targets)

        target_pairs_user = torch.cat(target_pairs)[types == 0]
        target_styles_user = torch.cat(target_styles)[types == 0]

        target_pairs_shop = torch.cat(target_pairs)[types == 1]
        target_styles_shop = torch.cat(target_styles)[types == 1]

        gts = torch.zeros(len(target_pairs_user), len(target_pairs_shop), dtype=torch.int64).to(logits.device)
        for i in range(len(target_pairs_user)):
            for j in range(len(target_pairs_shop)):
                tpu = target_pairs_user[i]
                tps = target_pairs_shop[j]
                tsu = target_styles_user[i]
                tss = target_styles_shop[j]
                if tps == tpu and tsu == tss and tss != 0 and tsu != 0:
                    gts[i, j] = 1
                else:
                    gts[i, j] = 0

        gts = gts.view(-1)

        # idx1 = torch.where(gts == 1)[0]
        # idx0 = torch.where(gts == 0)[0]
        # idx0 = torch.randperm(idx0.size(0))[:idx1.size(0)*2].to(idx1.device)
        #
        # keep_idxs = torch.cat([idx0, idx1], dim=0)
        logits = logits.view(-1, 2)
        loss = self.criterion(logits, gts)

        # loss = self.criterion(logits[keep_idxs, :], gts[keep_idxs])
        if loss > 1.0:
            # print("Dimezzo!")
            loss = loss / 2.0
            # loss = - torch.mean(
            #     gts * torch.log(logits[..., 1]) + (1 - gts) * torch.log(logits[..., 0]))

        return loss