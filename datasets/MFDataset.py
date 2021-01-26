import os
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
import random
import json


class MovingFashionDataset(Dataset):

    def __init__(self, jsonpath, transform=None, noise=True, root="", blacklist=None, whitelist=None):
        with open(jsonpath, "r") as fp:
            self.data = json.load(fp)
        if blacklist is not None:
            self.product_ids = sorted([k for k in self.data.keys() if k not in blacklist])
        else:
            if whitelist is not None:
                self.product_ids = sorted([k for k in self.data.keys() if k  in whitelist])
            else:
                self.product_ids = sorted([k for k in self.data.keys()])
        self.product_list = [self.data[k] for k in self.product_ids]
        self.noise = noise
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.product_list)

    def __getitem__(self, x):
        if isinstance(x, int):
            i = x
            tag, index = None, None
        elif isinstance(x, tuple):
            if len(x) == 3:
                i, tag, index = x
                video_i = None
            else:
                i, tag, index, video_i = x
        ret = {}
        ret["paths"] = self.product_list[i]
        ret["i"] = i
        tmp_paths = self.product_list[i]
        ret["video_i"] = -1
        if tag == "video":
            video_paths = tmp_paths["video_paths"]
            if video_i is None:
                video_name = random.choice(video_paths)
                ret["video_i"] = video_paths.index(video_name)
            else:
                ret["video_i"] = video_i
                video_name = video_paths[ret["video_i"]]
            video = cv2.VideoCapture(os.path.join(self.root, video_name))
            if isinstance(index, int):
                # if int you should find a value between 0.0 and 1.0 (index / fps * videolen)
                assert False
            n_frames = video.get(7)
            index2 = int(n_frames * index)
            video.set(1, index2)
            success, image = video.read()
            ret['valid'] = success
            # assert success
            # from cv2 to PIL
            if success:
                image = image[:, :, ::-1]
                tmp_noise = 0.25 if random.random() > 0.75 else 0.05
                if self.noise:
                    image = image / 255.0
                    image += np.random.randn(*image.shape) * tmp_noise
                    image = image * 255.0
                    image = np.clip(image, 0, 255.0)
                    image = np.asarray(image, dtype=np.uint8)
                img = Image.fromarray(image)
                if self.noise:
                    # img = img.resize((image.shape[1] // 3, image.shape[0] // 3))
                    img = img.resize((image.shape[1] // 2, image.shape[0] // 2))
            else:
                img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
            ret['index2'] = index2
            video.release()
        else:
            tmp_path = tmp_paths["img_path"]
            img = Image.open(os.path.join(self.root, tmp_path))
        ret['source'] = tmp_paths["source"]
        # ret['img'] = np.asarray(img)
        ret['index'] = index
        ret['tag'] = 1 if tag  != "video" else 0
        ret['labels'] = torch.tensor([0])
        ret['boxes'] = torch.tensor([[0.0, 0.0, img.size[0], img.size[1]]], dtype=torch.float32)
        ret['masks'] = torch.ones(1, img.size[1], img.size[0], dtype=torch.uint8)
        if self.transform is not None:
            img, ret = self.transform(img, ret)
        return img, ret

def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(dataset, batch_size, is_parallel, n_products=1
                   , first_n_withvideo=None, uniform_sampling=False, fixed_frame=None
                   , is_seq=False, num_workers=8, fixed_ind=None, fixed_video_i=None):
    if is_parallel:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        if is_seq:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)

    batch_sampler = NAPBatchSampler(dataset, sampler, batch_size, drop_last=True, n_products=n_products
                                    , first_n_withvideo=first_n_withvideo, uniform_sampling=uniform_sampling
                                    , fixed_frame=fixed_frame, fixed_ind=fixed_ind, fixed_video_i=fixed_video_i)

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler,
                                              collate_fn=collate_fn)
    # print("%d %d" % (rank, len(list(data_loader))))
    return data_loader


class NAPBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, sampler, batch_size, drop_last, n_samples=100
                 , n_products=1, first_n_withvideo=None, uniform_sampling=False, fixed_frame=None
                 , fixed_ind=None, fixed_video_i=None):
        super(NAPBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.data = dataset
        self.n_video_samples = n_samples
        self.n_products = n_products
        self.first_n_withvideo = first_n_withvideo
        self.uniform_sampling = uniform_sampling
        self.fixed_frame = fixed_frame
        self.fixed_ind = fixed_ind
        self.fixed_video_i = fixed_video_i

    def __iter__(self):
        batch = []
        count = -1
        for idx in self.sampler:
            if self.fixed_ind is not None:
                idx = self.fixed_ind
            batch.append((idx, "in", None))
            count += 1
            if self.batch_size == 1:
                tmp_video_samples = [x for x in np.linspace(0.0, 1.0, self.n_video_samples + 1)][:-1]
            else:
                if not self.uniform_sampling:
                    if self.fixed_frame is None:
                        tmp_video_samples = sorted([random.random()
                                                    for _ in range((self.batch_size // self.n_products) - 1)])
                        # tmp_video_samples = sorted(random.choices([0.5, 0.5], k=(self.batch_size // self.n_products) - 1))
                    else:
                        if isinstance(self.fixed_frame, list):
                            tmp_video_samples = [x for x in self.fixed_frame]
                        else:
                            tmp_video_samples = [self.fixed_frame for _ in range((self.batch_size // self.n_products) - 1)]
                else:
                    tmp_video_samples = [x for x in np.linspace(0.00, 1.0, (self.batch_size // self.n_products) - 1)]
            if self.first_n_withvideo is None or count < self.first_n_withvideo:
                for t in tmp_video_samples:
                    if self.fixed_video_i is None:
                        batch.append((idx, "video", t))
                    else:
                        batch.append((idx, "video", t, self.fixed_video_i))
            if self.batch_size == 1 or len(batch) == self.batch_size \
                    or self.first_n_withvideo is not None:
                yield batch
                batch = []
        if not self.drop_last:
            yield batch
            batch = []

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.n_products
        else:
            return 1 + (len(self.sampler) // self.n_products)
