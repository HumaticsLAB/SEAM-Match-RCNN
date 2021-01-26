from .match_batch_sampler import MatchingSampler
from .distributed import DistributedSampler
from .random import RandomSampler
import torch
from stuffs import utils


def get_dataloader(dataset, batch_size, is_parallel, n_products=0):
    if is_parallel:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset)

    batch_sampler = MatchingSampler(dataset, sampler, batch_size, drop_last=True, n_products=n_products)

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_sampler=batch_sampler,
                                              collate_fn=utils.collate_fn)
    # print("%d %d" % (rank, len(list(data_loader))))
    return data_loader
