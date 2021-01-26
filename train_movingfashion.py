from stuffs import transform as T
import torch
from models.maskrcnn import videomatchrcnn_resnet50_fpn
from stuffs.engine import train_one_epoch_videomf
import os
import torch.distributed as dist
import argparse
from datasets.MFDataset import MovingFashionDataset, get_dataloader
from evaluate_mf_video import evaluate
from torch.utils.tensorboard import SummaryWriter

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16384, rlimit[1]))

# DistributedDataParallel tutorial @ https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
# run with python -m torch.distributed.launch --nproc_per_node #GPUs train.py


gpu_map = [0, 1, 2, 3]

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# how many frames to extract from the video of each product


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
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
        device = torch.device("cuda")

    # DATASET ----------------------------------------------------------------------------------------------------------
    train_dataset = MovingFashionDataset(args.train_annots
                                  , transform=get_transform(True), noise=args.noise
                                  , root=args.root)
    test_dataset = MovingFashionDataset(args.test_annots
                                 , transform=T.ToTensor(), noise=args.noise
                                 , root=args.root)

    # ------------------------------------------------------------------------------------------------------------------

    # DATALOADER--------------------------------------------------------------------------------------------------------

    data_loader = get_dataloader(train_dataset, batch_size=args.batch_size_train
                                 , is_parallel=distributed, n_products=args.n_shops, num_workers=args.n_workers)
    data_loader_test = get_dataloader(test_dataset, batch_size=args.batch_size_test, is_parallel=distributed, num_workers=args.n_workers)

    # ------------------------------------------------------------------------------------------------------------------

    # MODEL ------------------------------------------------------------------------------------------------------------

    model = videomatchrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=14
                                        , n_frames=3)



    savefile = torch.load(args.pretrained_path)
    sd = savefile['model_state_dict']
    sd = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
    model.load_saved_matchrcnn(sd)
    model.to(device)
    # ------------------------------------------------------------------------------------------------------------------

    # OPTIMIZER AND SCHEDULER ------------------------------------------------------------------------------------------

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer
                                                        , milestones=args.milestones
                                                        , gamma=0.1)

    # ------------------------------------------------------------------------------------------------------------------


    if rank == 0:
        writer = SummaryWriter(os.path.join(args.save_path, args.save_tag))
    else:
        writer = None

    best_single, best_avg, best_aggr = 0.0, 0.0, 0.0

    for epoch in range(args.num_epochs):
        # train for one epoch, printing every 10 iterations
        print("Starting epoch %d for process %d" % (epoch, rank))
        train_one_epoch_videomf(model, optimizer, data_loader, device, epoch
                                 , print_freq=args.print_freq, score_thresh=0.1, writer=writer, inferstep=15)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        if rank == 0 and ((epoch % args.save_epochs) == 0):
            os.makedirs(args.save_path, exist_ok=True)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict().detach().cpu(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()
                }, os.path.join(args.save_path, (args.save_tag + "_epoch%03d") % epoch))
            model = model.to(device)

        if rank == 0 and ((epoch % args.eval_freq) == 0):
            model.eval()
            res = evaluate(model, data_loader_test, device, frames_per_product=args.frames_per_shop_test)
            writer.add_scalar("single_acc", res[0], global_step=epoch)
            writer.add_scalar("avg_acc", res[1], global_step=epoch)
            writer.add_scalar("aggr_acc", res[2], global_step=epoch)
            best_single, best_avg, best_aggr = max(res[0], best_single), max(res[1], best_avg)\
                                            , max(res[2], best_aggr)
            print("Best results:\n  - Best single: %01.2f"
                  "\n  - Best avg: %01.2f\n  - Best aggr: %01.2f\n" % (best_single, best_avg, best_aggr))

    os.makedirs(args.save_path, exist_ok=True)
    torch.save({
                    'epoch': args.num_epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()
            }, os.path.join(args.save_path, (args.save_tag + "_epoch%03d") % args.num_epochs))
    print("That's it!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="2,3")
    parser.add_argument("--n_workers", type=int, default=8)

    parser.add_argument("--frames_per_shop_train", type=int, default=10)
    parser.add_argument("--frames_per_shop_test", type=int, default=10)
    parser.add_argument("--n_shops", type=int, default=16)
    parser.add_argument("--root", type=str, default="/home/cjoppi/Documenti/CVPR2021/MovingFashion")
    parser.add_argument("--train_annots", type=str, default="/home/cjoppi/Documenti/CVPR2021/MovingFashion/train.json")
    parser.add_argument("--test_annots", type=str, default="/home/cjoppi/Documenti/CVPR2021/MovingFashion/test.json")
    parser.add_argument("--noise", type=bool, default=True)

    parser.add_argument("--num_epochs", type=int, default=31)
    parser.add_argument("--milestones", type=int, default=[15, 25])
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--pretrained_path", type=str, default="/media/data/mgodi/match_rcnn_clean/weights/df2matchrcnn")

    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--eval_freq", type=int, default=4)
    parser.add_argument("--save_epochs", type=int, default=2)

    parser.add_argument('--save_path',type=str, default="/media/data/cjoppi/CVPR2021/SEAM/models")
    parser.add_argument('--save_tag', type=str, default="NLB")

    args = parser.parse_args()

    args.batch_size_train = (1 + args.frames_per_shop_train) * args.n_shops
    args.batch_size_test = (1 +  args.frames_per_shop_test) * 1

    train(args)