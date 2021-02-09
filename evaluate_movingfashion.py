import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from datasets.MFDataset import MovingFashionDataset, get_dataloader
from models.video_maskrcnn import videomatchrcnn_resnet50_fpn
from stuffs import transform as T


def evaluate(model, data_loader, device, strategy="best_match"
             , score_threshold=0.1, k_thresholds=[1, 5, 10, 20]
             , frames_per_product=3, tracking_threshold=0.0, first_n_withvideo=None):
    count_products = 0
    count_street = 0
    shop_descrs = []
    street_descrs = []
    street_aggr_feats = []
    w = None
    b = None
    temporal_aggregator = model.roi_heads.temporal_aggregator
    for images, targets in tqdm(data_loader):
        count_products += 1
        images = list(image.to(device) for image in images)
        with torch.no_grad():
            step = 11
            output = [model(images[x:x + step]) for x in range(0, len(images), step)]
            output = [y for x in output for y in x]

        if not any(output[0]["scores"] >= score_threshold):
            continue
        if w is None:
            w = output[0]["w"].detach().cpu().numpy()
            b = output[0]["b"].detach().cpu().numpy()
        # take the best box as the product
        tmp_bs = output[0]["boxes"][output[0]["scores"] >= score_threshold]
        maxind = ((tmp_bs[:, 2] - tmp_bs[:, 0]) * (tmp_bs[:, 3] - tmp_bs[:, 1])).argmax().item()
        tmp_descr = temporal_aggregator(output[0]['roi_features'][maxind].unsqueeze(0)
                                        , torch.IntTensor([1]).to(device)
                                        , torch.LongTensor([0]).to(device))[1].detach().cpu().numpy()
        shop_descrs.append((output[0]['match_features'][maxind].detach().cpu().numpy()
                            , count_products - 1, tmp_descr, targets[0]["source"], targets[0]["i"])
                           )

        if first_n_withvideo is not None and count_products >= first_n_withvideo:
            continue

        count_street += 1

        current_start = len(street_descrs)
        tmp_roi_feats = []
        for i, o in enumerate(output[1:]):
            if any(o['scores'] >= score_threshold):
                for j in (o["scores"] >= score_threshold).nonzero().view(-1):
                    street_descrs.append((o['match_features'][j].detach().cpu().numpy()
                                          , count_products - 1
                                          , i
                                          , int(j.detach().cpu())
                                          , float(o["scores"][j].detach().cpu())
                                          , o["boxes"][j].detach().cpu()
                                          ,
                                          ))
                    tmp_roi_feats.append(o['roi_features'][j].unsqueeze(0))
        current_end = len(street_descrs)
        current_street_descrs = street_descrs[current_start:current_end]
        street_mat = np.concatenate([x[0][np.newaxis] for x in current_street_descrs])
        tmp_roi_feats = torch.cat(tmp_roi_feats, 0)
        aggr_feats = temporal_aggregator(tmp_roi_feats.to(device)
                                         , torch.IntTensor([0 for x in range(tmp_roi_feats.shape[0])]).to(device)
                                         , torch.LongTensor([0 for x in range(tmp_roi_feats.shape[0])])
                                         )[3][1:]
        aggr_feats = aggr_feats.view(-1, aggr_feats.shape[-1]).detach().cpu().numpy()
        street_aggr_feats.append(aggr_feats)


    torch.cuda.empty_cache()
    print(f"COUNT STREET: {count_street}")

    shop_mat = np.concatenate([x[0][np.newaxis].astype(np.float16) for x in shop_descrs])
    shop_prods = np.asarray([x[1] for x in shop_descrs])
    shop_sources = np.asarray([x[3] for x in shop_descrs])
    shop_datais = np.asarray([x[4] for x in shop_descrs])
    street_mat = np.concatenate([x[0][np.newaxis].astype(np.float16) for x in street_descrs])
    street_prods = np.asarray([x[1] for x in street_descrs])
    street_imgs = np.asarray([x[2] for x in street_descrs])
    street_scores = np.asarray([x[4] for x in street_descrs])
    street_aggr_feats = np.concatenate([x.astype(np.float16) for x in street_aggr_feats])
    shop_aggregated_descrs = np.concatenate([x[2][np.newaxis].astype(np.float16) for x in shop_descrs]).squeeze()


    def compute_ranking(inds):
        sq_diffs = (shop_mat[np.newaxis] - street_mat[inds, np.newaxis]) ** 2
        match_scores_raw = sq_diffs @ w.transpose().astype(np.float16) + b.astype(np.float16)
        match_scores_cls = np.exp(match_scores_raw) / np.exp(match_scores_raw).sum(2)[:, :, np.newaxis]
        match_scores = match_scores_cls[:, :, 1]
        match_rankings = np.argsort(match_scores, 1)[:, ::-1]
        return match_rankings

    def compute_distances(inds):
        sq_diffs = (shop_mat[np.newaxis] - street_mat[inds, np.newaxis]) ** 2
        match_scores_raw = sq_diffs @ w.transpose().astype(np.float16) + b.astype(np.float16)
        match_scores_cls = np.exp(match_scores_raw) / np.exp(match_scores_raw).sum(2)[:, :, np.newaxis]
        match_scores = match_scores_cls[:, :, 1]
        return match_scores

    def compute_raw_distances(inds):
        sq_diffs = (shop_mat[np.newaxis] - street_mat[inds, np.newaxis]) ** 2
        match_scores_raw = sq_diffs @ w.transpose().astype(np.float16) + b.astype(np.float16)
        match_scores = match_scores_raw[:, :, 1]
        return match_scores

    def compute_selfdist(inds):
        sq_diffs = (street_mat[np.newaxis, inds] - street_mat[inds, np.newaxis]) ** 2
        match_scores_raw = sq_diffs @ w.transpose().astype(np.float16) + b.astype(np.float16)
        # match_scores_cls = np.exp(match_scores_raw) / np.exp(match_scores_raw).sum(2)[:, :, np.newaxis]
        # match_scores = match_scores_cls[:, :, 1]
        match_scores = match_scores_raw[:, :, 1]
        return match_scores

    # calcolo i match score aggregati
    aggrW = temporal_aggregator.last.weight.detach().cpu().numpy().astype(np.float16)
    aggrB = temporal_aggregator.last.bias.detach().cpu().numpy().astype(np.float16)


    # performance table
    # N methods x K
    perf = np.zeros((8, len(k_thresholds)))

    # calcolo performance match normale
    k_accs = [0] * len(k_thresholds)
    k_accs_avg_desc = [0] * len(k_thresholds)
    k_accs_aggr_desc = [0] * len(k_thresholds)
    k_accs_avg_dist = [0] * len(k_thresholds)
    k_accs_max_dist = [0] * len(k_thresholds)
    k_accs_max_score = [0] * len(k_thresholds)
    total_querys = count_street * frames_per_product

    k_accs_reg = [0] * len(k_thresholds)
    k_accs_hard = [0] * len(k_thresholds)
    k_accs_avg_desc_reg = [0] * len(k_thresholds)
    k_accs_avg_desc_hard = [0] * len(k_thresholds)
    k_accs_aggr_desc_reg = [0] * len(k_thresholds)
    k_accs_aggr_desc_hard = [0] * len(k_thresholds)
    k_accs_max_dist_reg = [0] * len(k_thresholds)
    k_accs_max_dist_hard = [0] * len(k_thresholds)
    k_accs_avg_dist_reg = [0] * len(k_thresholds)
    k_accs_avg_dist_hard = [0] * len(k_thresholds)
    k_accs_max_score_reg = [0] * len(k_thresholds)
    k_accs_max_score_hard = [0] * len(k_thresholds)
    count_reg = 0
    count_hard = 0

    accs_per_product = {}

    track_lens = []

    all_ranks_list = []
    for p_i in tqdm(range(count_street)):
        if p_i in shop_prods:
            shop_prod_index = int((shop_prods == p_i).nonzero()[0][0])
            street_prod_indexes = (street_prods == p_i).nonzero()[0]
            unique_imgs = np.unique(street_imgs[street_prod_indexes])
            source = shop_sources[shop_prod_index]
            datakey = data_loader.dataset.product_ids[shop_datais[shop_prod_index]]

            ## TRACKING
            all_street_inds = (street_prods == p_i).nonzero()[0]
            tracking_simmat = compute_selfdist(all_street_inds)
            tracking_imgs = street_imgs[street_prod_indexes]
            best_match_ind = compute_raw_distances(all_street_inds)[:, p_i].argmax()
            tracking_slice = tracking_simmat[best_match_ind]
            track_inds = []
            track_imgs = []
            track_scores = []
            for i, ii in enumerate(unique_imgs):
                tmp_box_inds = (tracking_imgs == ii).nonzero()[0]
                tmp_box_inds2 = ((street_prods == p_i) & (street_imgs == ii)).nonzero()[0]
                tmp_score = tracking_slice[tmp_box_inds].max()
                if tmp_box_inds.size > 0 and tmp_score > tracking_threshold:
                    track_inds.append(tmp_box_inds2[tracking_slice[tmp_box_inds].argmax()])
                    track_imgs.append(ii)
                    track_scores.append(tmp_score)

            track_lens.append(len(track_inds))
            track_inds = np.asarray(track_inds)
            track_imgs = np.asarray(track_imgs)


            if source == 1:
                count_reg += 1
            else:
                count_hard += 1
            ranks_list = []
            best_inds = []
            distances = []
            scores = []
            accs_per_product[datakey] = {"sfmr": [0] * len(k_thresholds), "seamrcnn": [0] * len(k_thresholds)}
            for i, ii in enumerate(unique_imgs):
                if (track_imgs == ii).sum() > 0:
                    tmp_track_ind = track_inds[(track_imgs == ii).nonzero()[0]]
                    tmp_best_rank = (compute_ranking(tmp_track_ind) == shop_prod_index).nonzero()[1]
                    best_inds.append(tmp_track_ind[0])
                    ranks_list.append(tmp_best_rank)
                    for j, k in enumerate(k_thresholds):
                        if tmp_best_rank < k:
                            k_accs[j] += 1
                            accs_per_product[datakey]["sfmr"][j] += 1
                            if source == 1:
                                k_accs_reg[j] += 1
                            else:
                                k_accs_hard[j] += 1

                    distances.append(compute_distances(tmp_track_ind)[0])
                    scores.append(street_scores[tmp_track_ind])


            all_ranks_list.extend(ranks_list)
            best_inds = np.asarray(best_inds)

            # AGGR DESC
            seq_descs = torch.from_numpy(street_aggr_feats[best_inds]).unsqueeze(1).to(device)
            seq_mask = torch.zeros((1, 1 + seq_descs.shape[0]), device=seq_descs.device, dtype=torch.bool)
            new_seq_descs = torch.zeros((1 + seq_descs.shape[0], 1, 256)
                                        , device=seq_descs.device, dtype=seq_descs.dtype, requires_grad=False)
            new_seq_descs[1:] = seq_descs
            aggr_desc = temporal_aggregator(None, None, None
                                            , x3_1_seq=new_seq_descs.to(torch.float32)
                                            , x3_1_mask=seq_mask
                                            , x3_2=torch.from_numpy(shop_aggregated_descrs[shop_prod_index])
                                            .to(device).to(torch.float32))[0][0].detach().cpu().numpy()
            sq_diffs = (shop_aggregated_descrs[np.newaxis] - aggr_desc[np.newaxis, np.newaxis]) ** 2
            tmp_aggr_match_scores_raw = sq_diffs @ aggrW.transpose() + aggrB
            tmp_aggr_match_scores_cls = np.exp(tmp_aggr_match_scores_raw) \
                                        / np.exp(tmp_aggr_match_scores_raw).sum(2)[:, :,np.newaxis]
            tmp_aggr_match_scores = tmp_aggr_match_scores_cls[:, :, 1]
            tmp_aggr_match_rankings = np.argsort(tmp_aggr_match_scores, 1)[:, ::-1]
            aggr_desc_rank = (tmp_aggr_match_rankings == shop_prod_index).nonzero()[1].item()
            for j, k in enumerate(k_thresholds):
                if aggr_desc_rank < k:
                    k_accs_aggr_desc[j] += 1
                    accs_per_product[datakey]["seamrcnn"][j] += 1
                    if source == 1:
                        k_accs_aggr_desc_reg[j] += 1
                    else:
                        k_accs_aggr_desc_hard[j] += 1

            # AVG DESC
            avg_desc = street_mat[best_inds].mean(0)
            sq_diffs = (shop_mat[np.newaxis] - avg_desc[np.newaxis, np.newaxis]) ** 2
            match_scores_raw = sq_diffs @ w.transpose().astype(np.float16) + b.astype(np.float16)
            match_scores_cls = np.exp(match_scores_raw) / np.exp(match_scores_raw).sum(2)[:, :, np.newaxis]
            match_scores_cls = match_scores_cls[:, :, 1]
            avg_match_scores = match_scores_cls[0]
            tmp_ranks = np.argsort(avg_match_scores)[::-1]
            avg_desc_rank = (tmp_ranks == shop_prod_index).nonzero()[0].item()
            for j, k in enumerate(k_thresholds):
                if avg_desc_rank < k:
                    k_accs_avg_desc[j] += 1
                    if source == 1:
                        k_accs_avg_desc_reg[j] += 1
                    else:
                        k_accs_avg_desc_hard[j] += 1

            # AVG & MAX DISTANCE
            distances = np.stack(distances)
            avg_distances = distances.mean(0)
            tmp_ranks = np.argsort(avg_distances)[::-1]
            avg_dist_rank = (tmp_ranks == shop_prod_index).nonzero()[0].item()
            for j, k in enumerate(k_thresholds):
                if avg_dist_rank < k:
                    k_accs_avg_dist[j] += 1
                    if source == 1:
                        k_accs_avg_dist_reg[j] += 1
                    else:
                        k_accs_avg_dist_hard[j] += 1
            max_distances = distances.max(0)
            tmp_ranks = np.argsort(max_distances)[::-1]
            max_dist_rank = (tmp_ranks == shop_prod_index).nonzero()[0].item()
            for j, k in enumerate(k_thresholds):
                if max_dist_rank < k:
                    k_accs_max_dist[j] += 1
                    if source == 1:
                        k_accs_max_dist_reg[j] += 1
                    else:
                        k_accs_max_dist_hard[j] += 1

            # MAX CONFIDENCE SCORE
            scores = np.asarray(scores)
            max_score_ind = best_inds[scores.argmax()][np.newaxis]
            tmp_ranks = (compute_ranking(max_score_ind) == shop_prod_index).nonzero()[1]
            tmp_best_rank = tmp_ranks.item()
            for j, k in enumerate(k_thresholds):
                if tmp_best_rank < k:
                    k_accs_max_score[j] += 1
                    if source == 1:
                        k_accs_max_score_reg[j] += 1
                    else:
                        k_accs_max_score_hard[j] += 1

            # PER PRODUCT RESULTS
            accs_per_product[datakey]["sfmr"] = np.asarray(accs_per_product[datakey]["sfmr"]) / frames_per_product
            accs_per_product[datakey]["seamrcnn"] = np.asarray(accs_per_product[datakey]["seamrcnn"]) / 1.0

    torch.save(accs_per_product, "accs_per_product.pth")

    for k, k_acc in zip(k_thresholds, k_accs):
        print("Top-%d Retrieval Accuracy: %1.4f" % (k, k_acc / total_querys))
    ret1 = k_accs[0] / total_querys
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_avg_desc):
        print("Top-%d Retrieval Accuracy Product Avg Desc: %1.4f" % (k, k_acc / count_street))
    ret2 = k_accs_avg_desc[0] / count_street
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_aggr_desc):
        print("Top-%d Retrieval Accuracy Product Aggr Desc: %1.4f" % (k, k_acc / count_street))
    ret3 = k_accs_aggr_desc[0] / count_street
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_avg_dist):
        print("Top-%d Retrieval Accuracy Product Avg Dist: %1.4f" % (k, k_acc / count_street))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_max_dist):
        print("Top-%d Retrieval Accuracy Product Max Dist: %1.4f" % (k, k_acc / count_street))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_max_score):
        print("Top-%d Retrieval Accuracy Product Max Score: %1.4f" % (k, k_acc / count_street))
    print("*" * 50)


    # **************************************************************
    print("\n\n\n Regular ONLY")
    total_querys_reg = count_reg * frames_per_product

    for k, k_acc in zip(k_thresholds, k_accs_reg):
        print("Top-%d Retrieval Accuracy: %1.4f" % (k, k_acc / total_querys_reg))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_avg_desc_reg):
        print("Top-%d Retrieval Accuracy Product Avg Desc: %1.4f" % (k, k_acc / count_reg))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_aggr_desc_reg):
        print("Top-%d Retrieval Accuracy Product Aggr Desc: %1.4f" % (k, k_acc / count_reg))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_avg_dist_reg):
        print("Top-%d Retrieval Accuracy Product Avg Dist: %1.4f" % (k, k_acc / count_reg))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_max_dist_reg):
        print("Top-%d Retrieval Accuracy Product Max Dist: %1.4f" % (k, k_acc / count_reg))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_max_score_reg):
        print("Top-%d Retrieval Accuracy Product Max Score: %1.4f" % (k, k_acc / count_reg))
    print("*" * 50)

    # **************************************************************
    print("\n\n\n Hard ONLY")
    total_querys_hard = count_hard * frames_per_product

    for k, k_acc in zip(k_thresholds, k_accs_hard):
        print("Top-%d Retrieval Accuracy: %1.4f" % (k, k_acc / total_querys_hard))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_avg_desc_hard):
        print("Top-%d Retrieval Accuracy Product Avg Desc: %1.4f" % (k, k_acc / count_hard))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_aggr_desc_hard):
        print("Top-%d Retrieval Accuracy Product Aggr Desc: %1.4f" % (k, k_acc / count_hard))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_avg_dist_hard):
        print("Top-%d Retrieval Accuracy Product Avg Dist: %1.4f" % (k, k_acc / count_hard))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_max_dist_hard):
        print("Top-%d Retrieval Accuracy Product Max Dist: %1.4f" % (k, k_acc / count_hard))
    print("*" * 50)

    for k, k_acc in zip(k_thresholds, k_accs_max_score_hard):
        print("Top-%d Retrieval Accuracy Product Max Score: %1.4f" % (k, k_acc / count_hard))
    print("*" * 50)




    all_ranks_list = np.asarray(all_ranks_list)
    rm = np.median(all_ranks_list)
    rmq1 = np.percentile(all_ranks_list, 25)
    rmq3 = np.percentile(all_ranks_list, 75)
    print(f"Rank median: {rm}; rank 1st quartile: {rmq1}; rank 3rd quartile: {rmq3}")

    track_lens = np.asarray(track_lens)
    atl = float(track_lens.mean())
    print(f"Average Track Length: {atl}")

    perf[0] = np.asarray(k_accs, dtype=np.float32) / total_querys
    perf[1] = np.asarray(k_accs_avg_desc, dtype=np.float32) / count_street
    perf[2] = np.asarray(k_accs_aggr_desc, dtype=np.float32) / count_street

    import time
    perf = perf * 100
    os.makedirs("logs_mf", exist_ok=True)
    np.savetxt(os.path.join("logs_mf", str(time.time()) + ".csv"), perf, fmt="%02.2f", delimiter="\t")

    return ret1, ret2, ret3



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Testing")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="2,3")
    parser.add_argument("--n_workers", type=int, default=8)

    parser.add_argument("--frames_per_shop_test", type=int, default=10)
    parser.add_argument("--first_n_withvideo", type=int, default=100)
    parser.add_argument("--fixed_frame", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=0.0)

    parser.add_argument("--root", type=str, default="data/MovingFashion")
    parser.add_argument("--test_annots", type=str, default="data/MovingFashion/test.json")
    parser.add_argument("--noise", type=bool, default=True)

    parser.add_argument('--ckpt_path',type=str, default="ckpt/SEAM/MovingFashion/MF_epoch031")

    args = parser.parse_args()

    args.batch_size = (1 +  args.frames_per_shop_test) * 1

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_map = [0, 1, 2, 3]

    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
        rank = args.local_rank
        print("Distributed training with %d processors. This is #%s"
              % (int(os.environ['WORLD_SIZE']), rank))
    else:
        distributed = False
        rank = 0
        print("Not distributed training")

    if distributed:
        os.environ['NCCL_BLOCKING_WAIT'] = "1"
        torch.cuda.set_device(gpu_map[rank])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device(gpu_map[0]) if torch.cuda.is_available() else torch.device('cpu')

    test_dataset = MovingFashionDataset(args.test_annots
                                        , transform=T.ToTensor(), noise=args.noise
                                        , root=args.root)

    data_loader_test = get_dataloader(test_dataset, batch_size=args.batch_size_test, is_parallel=distributed,
                                      num_workers=args.n_workers)


    model = videomatchrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=14)


    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])

    model.to(device)
    model.eval()

    evaluate(model, data_loader_test, device, frames_per_product=args.frames_per_shop_test
             , first_n_withvideo=args.first_n_withvideo, score_threshold=args.score_threshold)
