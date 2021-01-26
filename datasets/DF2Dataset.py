import torchvision
import torch
from stuffs.mask_utils import annToMask
import torch.distributed as dist
import sys
import random
import numpy as np
from PIL import Image


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
            self, ann_file, root, transforms=None, noise=False, filter_onestreet=False
    ):
        super(DeepFashion2Dataset, self).__init__(root, ann_file)
        self.ids = sorted(self.ids)
        # # 63049, 63079, 134900, 134921, 139518, 139536, 133205, 133213
        # # 102172, 102176, 1503, 1513, 124558, 124565, 101325, 101338
        # ids = []
        # for img_id in self.ids:
        #     ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        #     anno = self.coco.loadAnns(ann_ids)
        #     if has_valid_annotation(anno):
        #         ids.append(img_id)
        # self.ids = ids
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
            self.street_match_keys = self.match_map_street.keys()
            for x in self.match_map_shop:
                if x not in self.street_match_keys:
                    to_del.append(x)
            for x in to_del:
                del self.match_map_shop[x]


        # print("Filtering images with no matches")
        # street_match_keys = self.match_map_street.keys()
        # shop_match_keys = self.match_map_shop.keys()
        # self.accepted_entries = []
        # for x in self.match_map_street:
        #     if x in shop_match_keys:
        #         self.accepted_entries = self.accepted_entries + self.match_map_street.get(x)
        #
        # for x in self.match_map_shop:
        #     if x in street_match_keys:
        #         self.accepted_entries = self.accepted_entries + self.match_map_shop.get(x)
        #
        # self.accepted_entries = list(set(self.accepted_entries))
        # print("Total images after filtering:" + str(len(self.accepted_entries)))

        self.noise = noise


    def __len__(self):
        return len(self.match_map_street)

    # TODO cambia la get per prendere prodotto invece che immagine

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

        img, anno = super(DeepFashion2Dataset, self).__getitem__(self.idx_to_id_map[idx])

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

        # filter crowd annotations
        # TODO might be better to add an extra field
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

        # if anno and "keypoints" in anno[0]:
        #     tmp = torch.stack([torch.as_tensor(obj['keypoints'], dtype=torch.float32) for obj in anno])
        #     keypoints = torch.zeros(tmp.size(0), 294, 3)
        #     keypoints[:, :, 0] = tmp[:, 0::3]
        #     keypoints[:, :, 1] = tmp[:, 1::3]
        #     keypoints[:, :, 2] = (tmp[:, 2::3] + 1) // 2
        #     target["keypoints"] = keypoints

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





class DeepFashion2DatasetEval(torchvision.datasets.coco.CocoDetection):
    def __init__(
            self, ann_file, root, transforms=None
    ):
        super(DeepFashion2DatasetEval, self).__init__(root, ann_file)
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


    def __getitem__(self, idx):
        img, anno = super(DeepFashion2DatasetEval, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
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

        # if anno and "keypoints" in anno[0]:
        #     tmp = torch.stack([torch.as_tensor(obj['keypoints'], dtype=torch.float32) for obj in anno])
        #     keypoints = torch.zeros(tmp.size(0), 294, 3)
        #     keypoints[:, :, 0] = tmp[:, 0::3]
        #     keypoints[:, :, 1] = tmp[:, 1::3]
        #     keypoints[:, :, 2] = (tmp[:, 2::3] + 1) // 2
        #     target["keypoints"] = keypoints

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
