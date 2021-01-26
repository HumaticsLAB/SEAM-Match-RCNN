import random
from torch.utils.data.sampler import BatchSampler
import torch
from torch._six import int_classes as _int_classes
from torch.utils.data import Sampler
import numpy as np


class MatchingSampler(Sampler):
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

    # def __iter__(self):
    #     batch = []
    #     for idx in self.sampler:
    #         ind = self.data.accepted_entries[idx]
    #         if ind in self.customer_inds:
    #             street_ind = ind
    #             shop_inds = self._getSamePairInShop(street_ind)
    #             if len(shop_inds) != 0:
    #                 # seed = self.seed_dict.get(street_ind)
    #                 # if seed is None: seed = 0
    #                 # random.seed(seed)
    #                 shop_ind = random.choice(shop_inds)
    #                 # self.shop_used[[i for i, x in enumerate(self.shop_inds) if x == shop_ind][0]] = 1
    #                 # self.customer_used[[i for i, x in enumerate(self.customer_inds) if x == street_ind][0]] = 1
    #                 batch.append(self.data.idx_to_id_map.get(street_ind))
    #                 batch.append(self.data.idx_to_id_map.get(shop_ind))
    #                 # self.seed_dict.update({street_ind: seed + 1})
    #                 self.tmp_index.append(str(shop_ind) + '_' + str(street_ind))
    #             else:
    #                 print(idx)
    #
    #         else:
    #             shop_ind = ind
    #             street_inds = self._getSamePairInStreet(shop_ind)
    #
    #             if len(street_inds) != 0:
    #                 # seed = self.seed_dict.get(shop_ind)
    #                 # if seed is None: seed = 0
    #                 # random.seed(seed)
    #                 street_ind = random.choice(street_inds)
    #                 # self.shop_used[[i for i, x in enumerate(self.shop_inds) if x == shop_ind][0]] = 1
    #                 # self.customer_used[[i for i, x in enumerate(self.customer_inds) if x == street_ind][0]] = 1
    #                 batch.append(self.data.idx_to_id_map.get(street_ind))
    #                 batch.append(self.data.idx_to_id_map.get(shop_ind))
    #                 # self.seed_dict.update({shop_ind: seed + 1})
    #                 self.tmp_index.append(str(shop_ind) + '_' + str(street_ind))
    #             else:
    #                 print(idx)
    #         if len(batch) == self.batch_size:
    #             yield batch
    #             batch = []
    #         if len(batch) > 0 and not self.drop_last:
    #             yield batch

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


class MatchingSamplerInfer(Sampler):

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
        self.customer_inds = self._getTypeInds('user')
        self.shop_inds = self._getTypeInds('shop')

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            if idx in self.customer_inds:
                batch = []
                batch.append(idx)
                for idx_s in self.shop_inds:
                    batch.append(idx_s)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = [idx]
                if len(batch) > 0 and not self.drop_last:
                    yield batch
            else:
                continue

    def __len__(self):
        if self.drop_last:
            return (len(self.shop_inds) // (self.batch_size - 1)) * len(self.customer_inds)
        else:
            return (len(self.shop_inds) // (self.batch_size - 1)) * len(self.customer_inds) + 1

    def _getTypeInds(self, type_s):
        inds = []
        N = len(self.data)
        for i in range(1, N + 1):
            if self.data.coco.imgs[i]['source'] == type_s:
                inds.append(i)

        return inds

    def _getSamePairInShop(self, id):
        match_desc = self.data.coco.imgs[id]['match_desc']

        for i in self.shop_inds:
            match_desc_s = self.data.coco.imgs[i]['match_desc']

            n_matches = len([(match_desc[k], match_desc_s[k]) for k in match_desc.keys() & match_desc_s.keys() if
                             match_desc[k] == match_desc_s[k]])
            if n_matches > 0:
                return i
        return -1
