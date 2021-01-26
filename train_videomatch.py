from stuffs import transform as T
import torch
from models.maskrcnn import videomatchrcnn_resnet50_fpn
from stuffs.engine import train_one_epoch_videonap
from torch import nn
from samplers import dataloader
import os
import torch.distributed as dist
import argparse
from datasets.NAPDataset import NAPDataset, get_dataloader, SimpleNAPDataset
from evaluate_nap_video import evaluate
from torch.utils.tensorboard import SummaryWriter
import time

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

# DistributedDataParallel tutorial @ https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
# run with python -m torch.distributed.launch --nproc_per_node #GPUs train.py

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# saveformat = "nograd_mtb_exp_0.25MA+A_newthresh_loss1_0.25_NLB_nopg1"
saveformat = "MA+A_-10thresh_0.3W_NLB_16x8_newstart_BEST_flip2"
# saveformat = "TEST"
# map rank to gpu
gpu_map = [0, 1, 2, 3]
split_path = "splits.pickle"
# how many frames to extract from the video of each product
frames_per_product_train = 8
# how big is the temporal aggregator
aggregator_size = 3
# how many products in the batch
n_products_batch = 16
batch_size_train = (1 + frames_per_product_train) * n_products_batch
frames_per_product_test = 10
batch_size_test = (1 + frames_per_product_test) * 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if 'WORLD_SIZE' in os.environ:
        # il distributed.launch setta il WORLD_SIZE a nproc_per_node
        # e spawna nproc_per_node processi passando a ciascuno un rank compreso fra 0 e nproc_per_node
        # il rank 0 è quello principale
        distributed = int(os.environ['WORLD_SIZE']) > 1
        rank = args.local_rank
        print("Distributed training with %d processors. This is #%s"
              % (int(os.environ['WORLD_SIZE']), rank))
    else:
        distributed = False
        rank = 0
        print("Not distributed training")

    if distributed:
        # dovrebbe far sì che i processi rispettino un timeout di 30 minuti per le varie operazioni di sync
        os.environ['NCCL_BLOCKING_WAIT'] = "1"
        # setto la GPU a i per il processo di rank i
        torch.cuda.set_device(gpu_map[rank])
        # questa operazione è bloccante, attende che tutti i processi siano pronti
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(torch.cuda.current_device())
    else:
        device = torch.device(gpu_map[0]) if torch.cuda.is_available() else torch.device('cpu')

    # DATASET ----------------------------------------------------------------------------------------------------------

    # if split_path is None or not os.path.isfile(split_path):
    #     tmp_dataset = NAPDataset("/media/data/mgodi/datasets/product_data.pt"
    #                              , dataset_root="/media/data/mgodi/datasets"
    #                              , transform=T.ToTensor())
    #     splits = tmp_dataset.get_split_inds()
    #     torch.save(splits, split_path)
    # splits = torch.load(split_path)
    # train_dataset = NAPDataset("/media/data/mgodi/datasets/product_data.pt"
    #                            , dataset_root="/media/data/mgodi/datasets"
    #                            , transform=T.ToTensor()
    #                            , split_inds=splits[0]
    #                            )
    # test_dataset = NAPDataset("/media/data/mgodi/datasets/product_data.pt"
    #                           , dataset_root="/media/data/mgodi/datasets"
    #                           , transform=T.ToTensor()
    #                           , split_inds=splits[1]
    #                           )
    train_dataset = SimpleNAPDataset("naptrain.json", transform=get_transform(True), noise=True)
    test_dataset = SimpleNAPDataset("naptest.json", transform=T.ToTensor(), noise=True)

    # ------------------------------------------------------------------------------------------------------------------

    # DATALOADER--------------------------------------------------------------------------------------------------------

    data_loader = get_dataloader(train_dataset, batch_size=batch_size_train
                                 , is_parallel=distributed, n_products=n_products_batch, num_workers=8)
    data_loader_test = get_dataloader(test_dataset, batch_size=batch_size_test, is_parallel=distributed, num_workers=8)

    # ------------------------------------------------------------------------------------------------------------------

    # MODEL ------------------------------------------------------------------------------------------------------------

    model = videomatchrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=14
                                        , n_frames=aggregator_size)
    model.to(device)

    # savefile_path = os.path.join("/media/data/mgodi/match_rcnn/saves/_epoch011")
    savefile_path = "/media/data/cjoppi/deepfashion2/models/TrainingAllImagesNoDecay/final_model"
    # savefile_path = "/media/data/mgodi/match_rcnn/saves_temporal/nograd_mtb_exp_MA+A_newthresh_loss1_0.25_OLA2_epoch048"
    savefile = torch.load(savefile_path)
    sd = savefile['model_state_dict']
    sd = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
    model.load_saved_matchrcnn(sd)
    # start_epoch = savefile['epoch']

    # savefile_path = "/media/data/mgodi/match_rcnn/saves_temporal/MA+A_-10thresh_0.3W_OLA_6x4_newstart_BEST_flip_epoch035"
    # savefile = torch.load(savefile_path)
    # sd = savefile['model'].state_dict()
    # model.load_state_dict(sd)

    # if distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_map[rank]]
    #                                                       , output_device=gpu_map[rank])

    # ------------------------------------------------------------------------------------------------------------------

    # OPTIMIZER AND SCHEDULER ------------------------------------------------------------------------------------------

    # construct an optimizer
    # pg0 = []
    # pg1 = []
    # for k, v in dict(model.named_parameters()).items():
    #     if not v.requires_grad:
    #         continue
    #     if "temporal_aggregator.newnlb" in k or "temporal_aggregator.nlb" in k:
    #         pg1 += [v]
    #     else:
    #         pg0 += [v]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.02,
                                momentum=0.9, weight_decay=0.0005)
    # optimizer.add_param_group({"params": pg1, "lr": 1.0})
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=4,
    #                                                gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer
                                                        , milestones=[15]
                                                        , gamma=0.1)

    # optimizer.load_state_dict(savefile['optimizer_state_dict'])
    # lr_scheduler.load_state_dict(savefile['scheduler_state_dict'])
    # lr_scheduler.milestones = [30]

    # ------------------------------------------------------------------------------------------------------------------

    num_epochs = 51
    save_epochs = 1
    save_path = 'saves_temporal/'

    writer = SummaryWriter(os.path.join(save_path, saveformat))

    best_single, best_avg, best_aggr = 0.0, 0.0, 0.0

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        print("Starting epoch %d for process %d" % (epoch, rank))
        train_one_epoch_videonap(model, optimizer, data_loader, device, epoch
                                 , print_freq=50, score_thresh=0.3, writer=writer, inferstep=20)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        if rank == 0 and ((epoch % 5) == 0):
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                "model": model.to("cpu")
            }, os.path.join(save_path, (saveformat + "_epoch%03d") % epoch))
            model = model.to(device)

        if rank == 0 and ((epoch % 5) == 0 and (epoch > 20)):
            model.eval()
            res = evaluate(model, data_loader_test, device, frames_per_product=frames_per_product_test
                           , first_n_withvideo=None)
            writer.add_scalar("single_acc", res[0], global_step=epoch)
            writer.add_scalar("avg_acc", res[1], global_step=epoch)
            writer.add_scalar("aggr_acc", res[2], global_step=epoch)
            best_single, best_avg, best_aggr = max(res[0], best_single), max(res[1], best_avg)\
                                            , max(res[2], best_aggr)
            print("Best results:\n  - Best single: %01.2f"
                  "\n  - Best avg: %01.2f\n  - Best aggr: %01.2f\n" % (best_single, best_avg, best_aggr))
            time.sleep(5)

    print("That's it!")
