import math
import random
import sys

import torch
import torch.distributed as dist
import torchvision
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler

from stuffs import utils
from stuffs.mask_utils import annToMask


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= 10:
        return True
    return False


class DeepFashion2Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, transforms=None
    ):
        super(DeepFashion2Dataset, self).__init__(root, ann_file)
        self.ids = sorted(self.ids)

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.idx_to_id_map = {v: k for k, v in enumerate(self.ids)}

        self._transforms = transforms
        self.street_inds = self._getTypeInds('user')
        self.shop_inds = self._getTypeInds('shop')

        self.match_map_shop = {}
        self.match_map_street = {}
        print("Computing Street Match Descriptors map")
        for i in self.street_inds:
            e = self.coco.imgs[i]
            for x in e['match_desc']:
                if x == '0':
                    continue
                hashable_key = x + '_' + str(e['match_desc'].get(x))
                inds = self.match_map_street.get(hashable_key)
                if inds is None:
                    self.match_map_street.update({hashable_key: [i]})
                else:
                    inds.append(i)
                    self.match_map_street.update({hashable_key: inds})
        print("Computing Shop Match Descriptors map")
        for i in self.shop_inds:
            e = self.coco.imgs[i]
            for x in e['match_desc']:
                if x == '0':
                    continue
                hashable_key = x + '_' + str(e['match_desc'].get(x))
                inds = self.match_map_shop.get(hashable_key)
                if inds is None:
                    self.match_map_shop.update({hashable_key: [i]})
                else:
                    inds.append(i)
                    self.match_map_shop.update({hashable_key: inds})

        print("Filtering images with no matches")
        street_match_keys = list(self.match_map_street.keys())
        shop_match_keys = self.match_map_shop.keys()
        self.accepted_entries = []
        for x in self.match_map_street:
            if x in shop_match_keys:
                self.accepted_entries = self.accepted_entries + self.match_map_street.get(x)

        for x in self.match_map_shop:
            if x in street_match_keys:
                self.accepted_entries = self.accepted_entries + self.match_map_shop.get(x)

        self.accepted_entries = list(set(self.accepted_entries))
        print("Total images after filtering:" + str(len(self.accepted_entries)))

    def __getitem__(self, idx):
        img, anno = super(DeepFashion2Dataset, self).__getitem__(idx)

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno if obj['area'] != 0]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

        classes = [obj["category_id"] for obj in anno if obj['area'] != 0]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)

        target = {}
        target["labels"] = classes
        target["boxes"] = boxes
        target["classes"] = classes

        if anno and "area" in anno[0]:
            area = torch.stack([torch.as_tensor(obj['area'], dtype=torch.float32) for obj in anno if obj['area'] != 0])
            target["area"] = area

        if anno and "segmentation" in anno[0]:
            masks = torch.stack(
                [torch.as_tensor(annToMask(obj, size=[img.height, img.width]), dtype=torch.uint8) for obj in anno if obj['area'] != 0])
            target["masks"] = masks

        if anno and "pair_id" in anno[0]:
            pair_ids = [obj['pair_id'] for obj in anno if obj['area'] != 0]
            pair_ids = torch.tensor(pair_ids)
            target["pair_ids"] = pair_ids


        if anno and "style" in anno[0]:
            styles = [obj['style'] for obj in anno if obj['area'] != 0]
            styles = torch.tensor(styles)
            target["styles"] = styles

        if anno and "source" in anno[0]:
            sources = [0 if obj['source'] == 'user' else 1 for obj in anno if obj['area'] != 0]
            # print("-->", idx, sources)
            sources = torch.tensor(sources)
            target["sources"] = sources

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, anno[0]['image_id']

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def _getTypeInds(self, type_s):
        inds = []
        N = len(self.coco.imgs)
        for i in self.ids:
            if self.coco.imgs[i]['source'] == type_s:
                inds.append(i)

        return inds


def get_dataloader(dataset, batch_size, is_parallel, num_workers=0):
    if is_parallel:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset)

    batch_sampler = DF2MatchingSampler(dataset, sampler, batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler,
                                              collate_fn=utils.collate_fn)
    return data_loader



class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source.accepted_entries

        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples




class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset.accepted_entries

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DF2MatchingSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

        Args:
            sampler (Sampler): Base sampler.
            batch_size (int): Size of mini-batch.
            drop_last (bool): If ``True``, the sampler will drop the last batch if
                its size would be less than ``batch_size``

        Example:
            >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
            >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """

    def __init__(self, dataset, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.data = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.customer_inds = self.data.street_inds
        self.shop_inds = self.data.shop_inds
        self.customer_used = torch.zeros((len(self.customer_inds, )))
        self.shop_used = torch.zeros((len(self.shop_inds, )))
        self.match_map_shop = self.data.match_map_shop
        self.match_map_street = self.data.match_map_street
        self.seed_dict = {}
        self.tmp_index = []

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            ind = self.data.accepted_entries[idx]
            if ind in self.customer_inds:
                street_ind = ind
                shop_inds = self._getSamePairInShop(street_ind)
                if len(shop_inds) != 0:
                    shop_ind = random.choice(shop_inds)
                    batch.append(self.data.idx_to_id_map.get(street_ind))
                    batch.append(self.data.idx_to_id_map.get(shop_ind))
                    self.tmp_index.append(str(shop_ind) + '_' + str(street_ind))
                else:
                    print(idx)

            else:
                shop_ind = ind
                street_inds = self._getSamePairInStreet(shop_ind)

                if len(street_inds) != 0:
                    street_ind = random.choice(street_inds)
                    batch.append(self.data.idx_to_id_map.get(street_ind))
                    batch.append(self.data.idx_to_id_map.get(shop_ind))
                    self.tmp_index.append(str(shop_ind) + '_' + str(street_ind))
                else:
                    print(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // (self.batch_size // 2)
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def _getTypeInds(self, type_s):
        inds = []
        N = len(self.data)
        for i in range(1, N + 1):
            if self.data.coco.imgs[i]['source'] == type_s:
                inds.append(i)
        return inds

    def _getSamePairInShop(self, id):
        match_desc = self.data.coco.imgs[id]['match_desc']
        ids = []
        for x in match_desc:
            hashable_key = x + '_' + str(match_desc.get(x))
            matches = self.match_map_shop.get(hashable_key)
            if matches is not None:
                ids = ids + matches
        return ids

    def _getSamePairInStreet(self, id):
        match_desc = self.data.coco.imgs[id]['match_desc']
        ids = []

        for x in match_desc:
            hashable_key = x + '_' + str(match_desc.get(x))
            matches = self.match_map_street.get(hashable_key)
            if matches is not None:
                ids = ids + matches

        return ids