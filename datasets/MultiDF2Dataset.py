import random
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torch._six import int_classes as _int_classes
from torch.utils.data import RandomSampler, DistributedSampler
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


class MultiDeepFashion2Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, transforms=None, noise=False, filter_onestreet=False
    ):
        super(MultiDeepFashion2Dataset, self).__init__(root, ann_file)
        self.ids = sorted(self.ids)

        print(len(self.ids))

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

        if filter_onestreet:
            print("Filtering products with one street or less")

            to_del = []
            self.shop_match_keys = self.match_map_shop.keys()
            for x in self.match_map_street:
                if x not in self.shop_match_keys or len(self.match_map_street[x]) < 2:
                    to_del.append(x)
            for x in to_del:
                del self.match_map_street[x]

            to_del = []
            self.street_match_keys = list(self.match_map_street.keys())
            for x in self.match_map_shop:
                if x not in self.street_match_keys:
                    to_del.append(x)
            for x in to_del:
                del self.match_map_shop[x]

        self.noise = noise


    def __len__(self):
        return len(self.match_map_street)


    def __getitem__(self, x):
        # i: product id
        # tag: "shop" or "street"
        # index: None if tag is "shop" else index of street
        i, tag, index = x
        if tag == "shop":
            idx = random.choice(self.match_map_shop[i])
        else:
            index2 = int(len(self.match_map_street[i]) * index)
            idx = self.match_map_street[i][index2]

        img, anno = super(MultiDeepFashion2Dataset, self).__getitem__(self.idx_to_id_map[idx])

        # ****************************************************
        image = np.array(img)
        if self.noise:
            tmp_noise = 0.1 if random.random() > 0.75 else 0.0
        else:
            tmp_noise = 0.0
        image = image / 255.0
        image += np.random.randn(*image.shape) * tmp_noise
        image = image * 255.0
        image = np.clip(image, 0, 255.0)
        image = np.asarray(image, dtype=np.uint8)
        img = Image.fromarray(image)
        # ****************************************************

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


        target["i"] = i
        target['tag'] = 1 if tag == "shop" else 0

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

def get_dataloader(dataset, batch_size, is_parallel, n_products=0, n_workers=0):
    if is_parallel:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = RandomSampler(dataset)

    batch_sampler = MultiDF2BatchSampler(dataset, sampler, batch_size, drop_last=True, n_products=n_products)

    data_loader = torch.utils.data.DataLoader(dataset, num_workers=n_workers, batch_sampler=batch_sampler,
                                              collate_fn=utils.collate_fn)
    return data_loader


class MultiDF2BatchSampler(Sampler):
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

    def __init__(self, dataset, sampler, batch_size, drop_last, n_products=0):
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
        self.n_products = n_products
        self.seed_dict = {}
        self.tmp_index = []
        pair_keys = [k for k in self.data.match_map_street.keys() if k in self.data.match_map_shop]
        pair_keys += [k for k in self.data.match_map_shop.keys() if k in self.data.match_map_street]
        self.pair_keys = list(set(pair_keys))

    def __iter__(self):
        batch = []
        count = -1
        pair_keys = self.pair_keys
        for idx in self.sampler:
            batch.append((pair_keys[idx], "shop", None))
            count += 1
            tmp_video_samples = sorted([random.random() for x in range((self.batch_size // self.n_products) - 1)])

            for t in tmp_video_samples:
                batch.append((pair_keys[idx], "street", t))
            if self.batch_size == 1 or len(batch) == self.batch_size:
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
