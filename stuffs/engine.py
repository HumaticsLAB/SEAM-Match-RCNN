import math
import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as maskUtils

from models.match_head import MatchLossWeak, NEWBalancedAggregationMatchLossWeak \
    , AggregationMatchLossDF2
from stuffs import utils

outputkeys_whitelist = ['scores', 'boxes', 'roi_features']


def train_one_epoch_matchrcnn(model, optimizer, data_loader, device, epoch, print_freq
                             , writer=None):
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        rank = dist.get_rank()
    else:
        rank = 0
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    count = -1
    for i, (images, targets, idxs) in enumerate(metric_logger.log_every(data_loader, print_freq, header, rank=rank)):
        count += 1
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        # print(args.local_rank)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if writer is not None and (((count % print_freq) == 0) or count == 0):
            global_step = (epoch * len(data_loader)) + count
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(k, v.item(), global_step=global_step)
            writer.add_scalar("loss", losses.item(), global_step=global_step)

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            print(idxs)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print("Epoch finished by process #%d" % rank)




def train_one_epoch_movingfashion(model, optimizer, data_loader, device, epoch, print_freq
                             , score_thresh=0.7, writer=None, inferstep=10):
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        distributed = True
        rank = dist.get_rank()
    else:
        distributed = False
        rank = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # wait for all workers to be ready
    # if distributed:
    #     dist.barrier()
    real_model = model if not hasattr(model, "module") else model.module
    match_predictor = real_model.roi_heads.match_predictor
    tracking_predictor = real_model.roi_heads.tracking_predictor
    temporal_aggregator = real_model.roi_heads.temporal_aggregator


    match_loss = MatchLossWeak(device)
    aggregation_loss = NEWBalancedAggregationMatchLossWeak(device, temporal_aggregator)

    count = -1
    for images, targets in metric_logger.log_every(data_loader, print_freq, header, rank=rank):

        count += 1
        images = list(image.to(device) for image in images)
        # targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        # output: list of dicts: "boxes", "labels", "scores", "masks", "match_features", "w", "b", "roi_features"
        model.eval()
        with torch.no_grad():
            output = [model(images[x:x + inferstep]) for x in range(0, len(images), inferstep)]
            output = [y for x in output for y in x]
        # clean output dict to save memory
        output = [{k: v for k, v in o.items() if k in outputkeys_whitelist} for o in output]

        match_predictor.train()
        tracking_predictor.train()
        temporal_aggregator.train()

        roi_features = []
        types = []
        prod_ids = []
        img_ids = []
        exclude_prod_ids = []
        boxes = []
        scores = []
        for i, (t, o) in enumerate(zip(targets, output)):
            if t["i"] in exclude_prod_ids:
                # if a product is excluded, skip all frames
                continue
            if "roi_features" in o:
                indexes = (o["scores"] >= score_thresh).nonzero().view(-1)
                if indexes.numel() < 1:
                    if t["tag"] == 1:
                        # exclude street imgs if shop doesn't have any boxes
                        exclude_prod_ids.append(t["i"])
                    continue
                if t["tag"] == 1:
                    tmp_bs = o["boxes"][indexes]
                    indexes = ((tmp_bs[:, 2] - tmp_bs[:, 0]) * (tmp_bs[:, 3] - tmp_bs[:, 1])).argmax().view(1)
                roi_features.append(o["roi_features"][indexes])
                boxes.append(o["boxes"][indexes])
                scores.append(o["scores"][indexes])
                types = types + [t["tag"]] * indexes.shape[0]
                prod_ids = prod_ids + [t["i"]] * indexes.shape[0]
                img_ids = img_ids + [i] * indexes.shape[0]
        flag = False
        types = torch.IntTensor(types)
        # at least two boxes, one being a street and one being a shop
        if len(roi_features) >= 2 and (types == 0).any() and (types == 1).any():
            roi_features = torch.cat(roi_features, 0)


            # predict matches street-shop
            _, logits = match_predictor(roi_features, types)
            # predict tracking street-street
            # first retrieve only the street items
            # duplicate them to match with each other
            weight_aggr = min(epoch / 1, 1.0)

            loss_dict = {
                'match_loss': match_loss(logits, types, prod_ids, img_ids)
                , 'aggregation_loss': weight_aggr *
                                      aggregation_loss(logits, types, prod_ids, img_ids, roi_features)
                 }

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        if writer is not None and (((count % print_freq) == 0) or count == 0):
            global_step = (epoch * len(data_loader)) + count
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(k, v.item(), global_step=global_step)
            writer.add_scalar("loss", losses.item(), global_step=global_step)


        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print("Epoch finished by process #%d" % rank)


def train_one_epoch_multiDF2(model, optimizer, data_loader, device, epoch, print_freq
                             , score_thresh=0.7, writer=None, inferstep=10, use_gt=False):
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        rank = dist.get_rank()
    else:
        rank = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    match_predictor = model.roi_heads.match_predictor
    tracking_predictor = model.roi_heads.tracking_predictor
    temporal_aggregator = model.roi_heads.temporal_aggregator
    aggregation_loss = AggregationMatchLossDF2(device, temporal_aggregator)

    count2 = -1
    for images, targets, ids in metric_logger.log_every(data_loader, print_freq, header, rank=rank):
        # if count2 >= 5:
        #     break
        count2 += 1
        images = list(image.to(device) for image in images)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        targets = [{k: (v.float() if k == "boxes" else v) for k, v in t.items()} for t in targets]
        # output: list of dicts: "boxes", "labels", "scores", "masks", "match_features", "w", "b", "roi_features"
        model.eval()
        targets2 = deepcopy(targets)
        with torch.no_grad():
            if use_gt:
                output = [model(images[x:x + inferstep], targets=targets2[x:x + inferstep]) for x in range(0, len(images), inferstep)]
            else:
                output = [model(images[x:x + inferstep]) for x in range(0, len(images), inferstep)]
            output = [y for x in output for y in x]
        # clean output dict to save memory
        output = [{k: v for k, v in o.items() if k in outputkeys_whitelist} for o in output]
        # torch.cuda.empty_cache()

        match_predictor.eval()
        tracking_predictor.eval()
        temporal_aggregator.train()
        # print(args.local_rank)

        roi_features = []
        types = []
        prod_ids = []
        img_ids = []
        gt_infos = []
        exclude_prod_ids = []
        boxes = []
        scores = []
        i2tmpid = {}
        count = 0
        for i, (t, o) in enumerate(zip(targets, output)):
            if t["i"] in exclude_prod_ids:
                # if a product is excluded, skip all frames
                continue
            if "roi_features" in o:
                indexes = (o["scores"] >= score_thresh).nonzero().view(-1)
                if indexes.numel() < 1:
                    if t["tag"] == 1:
                        # exclude street imgs if shop doesn't have any boxes
                        exclude_prod_ids.append(t["i"])
                    continue
                if t["i"] not in i2tmpid:
                    i2tmpid[t["i"]] = count
                    count += 1
                pr_boxes = o["boxes"][indexes].detach().cpu().numpy()
                gt_boxes = t["boxes"].detach().cpu().numpy()
                pr_boxes[:, 2] = pr_boxes[:, 2] - pr_boxes[:, 0]
                pr_boxes[:, 3] = pr_boxes[:, 3] - pr_boxes[:, 1]
                gt_boxes[:, 2] = gt_boxes[:, 2] - gt_boxes[:, 0]
                gt_boxes[:, 3] = gt_boxes[:, 3] - gt_boxes[:, 1]
                iou = maskUtils.iou(gt_boxes, pr_boxes, np.zeros((pr_boxes.shape[0]))) # gts x preds
                style, pair_id = [int(x) for x in t["i"].split("_")]
                gt_prods = [(count if (t["styles"][ind] == style and t["pair_ids"][ind] == pair_id) else -1)
                                                        for ind in range(gt_boxes.shape[0])]
                det_prods = [-1] * indexes.shape[0]
                det_prods[iou[torch.tensor(gt_prods).argmax()].argmax()] = count

                if t["tag"] == 1:
                    indexes = indexes[torch.tensor(det_prods) == count]
                    det_prods = [count]

                roi_features.append(o["roi_features"][indexes])
                boxes.append(o["boxes"][indexes])
                scores.append(o["scores"][indexes])
                types = types + [t["tag"]] * indexes.shape[0]
                prod_ids = prod_ids + [i2tmpid[t["i"]]] * indexes.shape[0]
                img_ids = img_ids + [i] * indexes.shape[0]
                gt_infos = gt_infos + det_prods
        types = torch.IntTensor(types)
        # at least two boxes, one being a street and one being a shop
        if len(roi_features) >= 2 and (types == 0).any() and (types == 1).any():
            roi_features = torch.cat(roi_features, 0)

            # predict matches street-shop
            _, logits = match_predictor(roi_features, types)

            weight_aggr = 1.0

            loss_dict = {
                'aggregation_loss': weight_aggr *
                                      aggregation_loss(types, roi_features, gt_infos)
                         }

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        if writer is not None and (((count % print_freq) == 0) or count == 0):
            global_step = (epoch * len(data_loader)) + count
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(k, v.item(), global_step=global_step)
            writer.add_scalar("loss", losses.item(), global_step=global_step)


        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print("Epoch finished by process #%d" % rank)
