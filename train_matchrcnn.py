from stuffs import transform as T
from datasets.DF2Dataset import DeepFashion2Dataset, get_dataloader
import torch
from models.matchrcnn import matchrcnn_resnet50_fpn
from stuffs.engine import train_one_epoch_matchrcnn
import os
import torch.distributed as dist
import argparse

from torch.utils.tensorboard import SummaryWriter

gpu_map = [0, 1, 2, 3]


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# run with python -m torch.distributed.launch --nproc_per_node #GPUs train_matchrcnn.py


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

    train_dataset = DeepFashion2Dataset(root=args.root_train
                                        , ann_file=args.train_annots,
                                        transforms=get_transform(True))

    # ------------------------------------------------------------------------------------------------------------------

    # DATALOADER--------------------------------------------------------------------------------------------------------

    data_loader = get_dataloader(train_dataset, batch_size=args.batch_size, is_parallel=distributed)

    # ------------------------------------------------------------------------------------------------------------------

    # MODEL ------------------------------------------------------------------------------------------------------------
    from models.maskrcnn import params as c_params
    model = matchrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=14, **c_params)
    model.to(device)

    # ------------------------------------------------------------------------------------------------------------------

    # OPTIMIZER AND SCHEDULER ------------------------------------------------------------------------------------------

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate,
                                momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)

    # ------------------------------------------------------------------------------------------------------------------

    if rank == 0:
        writer = SummaryWriter(os.path.join(args.save_path, args.save_tag))
    else:
        writer = None

    for epoch in range(args.num_epochs):
        # train for one epoch, printing every 10 iterations
        print("Starting epoch %d for process %d" % (epoch, rank))
        train_one_epoch_matchrcnn(model, optimizer, data_loader, device, epoch, args.print_freq, writer)
        # update the learning rate
        lr_scheduler.step()

        if rank == 0 and epoch != 0 and epoch % args.save_epochs == 0:
            os.makedirs(args.save_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, os.path.join(args.save_path, (args.save_tag + "_epoch%03d") % epoch))

    os.makedirs(args.save_path, exist_ok=True)
    torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
    }, os.path.join(args.save_path, "final_model"))

    print("That's it!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Match R-CNN Training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--n_workers", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--root_train", type=str, default='data/deepfashion2/train/image')
    parser.add_argument("--train_annots", type=str, default='data/deepfashion2/train/annots.json')

    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--milestones", type=int, default=[6, 9])
    parser.add_argument("--learning_rate", type=float, default=0.02)

    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_epochs", type=int, default=2)

    parser.add_argument('--save_path', type=str, default="ckpt/matchrcnn")
    parser.add_argument('--save_tag', type=str, default="DF2-pretraining")

    args = parser.parse_args()

    train(args)
