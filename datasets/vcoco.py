# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from torchvision.datasets.vision import VisionDataset
import torchvision
import torch
import numpy as np
import json
import cv2
import random
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as F
from util.box_ops import box_xyxy_to_cxcywh
from PIL import Image


coco_classes_originID = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}


coco_instance_ID_to_name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


hoi_interaction_names = json.loads(
    open('./data/vcoco/vcoco_verb_names.json', 'r').readlines()[0])['verb_names']


def convert_xywh2x1y1x2y2(box, shape, flip):
    ih, iw = shape[:2]
    x, y, w, h = box
    if flip == 1:
        x1_org = x
        x2_org = x + w - 1
        x2 = iw - 1 - x1_org
        x1 = iw - 1 - x2_org
    else:
        x1 = x
        x2 = x + w - 1
    x1 = max(x1, 0)
    x2 = min(x2, iw-1)
    y1 = max(y, 0)
    y2 = min(y + h - 1, ih-1)
    return [x1, y1, x2, y2]


def get_det_annotation_from_odgt(item, shape, flip, gt_size_min=1):
    total_boxes, gt_boxes, ignored_boxes = [], [], []
    for annot in item['gtboxes']:
        box = convert_xywh2x1y1x2y2(annot['box'], shape, flip)
        x1, y1, x2, y2 = box
        cls_id = coco_classes_originID[annot['tag']]
        total_boxes.append([x1, y1, x2, y2, cls_id, ])
        if annot['tag'] not in coco_classes_originID:
            continue
        if annot.get('extra', {}).get('ignore', 0) == 1:
            ignored_boxes.append(box)
            continue
        if (x2 - x1 + 1) * (y2 - y1 + 1) < gt_size_min ** 2:
            ignored_boxes.append(box)
            continue
        if x2 <= x1 or y2 <= y1:
            ignored_boxes.append(box)
            continue
        gt_boxes.append([x1, y1, x2, y2, cls_id, ])
    return gt_boxes, ignored_boxes, total_boxes


def get_interaction_box(human_box, object_box, hoi_id):
    hx1, hy1, hx2, hy2, hid = human_box
    ox1, oy1, ox2, oy2, oid = object_box
    # hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    # ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    # dx = (hcx - ocx) / 5
    # dy = (hcy - ocy) / 5
    # xx1, yy1, xx2, yy2 = list(map(int, [ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy]))
    xx1, yy1, xx2, yy2 = min(hx1, ox1), min(hy1, oy1), max(hx2, ox2), max(hy2, oy2)
    return [xx1, yy1, xx2, yy2, hoi_id]


def xyxy_to_cxcywh(box):
    x0, y0, x1, y1, cid = box
    return [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0), cid]


def get_hoi_annotation_from_odgt(item, total_boxes, scale):
    human_boxes, object_boxes, action_boxes = [], [], []
    human_labels, object_labels, action_labels = [], [], []
    img_hh, img_ww = item['height'], item['width']
    for hoi in item.get('hoi', []):
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[hoi['subject_id']]))
        human_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        x1, y1, x2, y2, cls_id = list(map(int, total_boxes[hoi['object_id']]))
        object_box = x1 // scale, y1 // scale, x2 // scale, y2 // scale, cls_id
        if cls_id == -1 or x1 >= x2 or y1 >= y2:
            continue
        hoi_id = hoi_interaction_names.index(hoi['interaction'])
        hoi_box = get_interaction_box(human_box=human_box, object_box=object_box, hoi_id=hoi_id)

        human_boxes.append(human_box[0:4])
        object_boxes.append(object_box[0:4])
        action_boxes.append(hoi_box[0:4])
        human_labels.append(human_box[4])
        object_labels.append(object_box[4])
        action_labels.append(hoi_box[4])
    return dict(
        human_boxes=torch.from_numpy(np.array(human_boxes).astype(np.float32)),
        human_labels=torch.from_numpy(np.array(human_labels)),
        object_boxes=torch.from_numpy(np.array(object_boxes).astype(np.float32)),
        object_labels=torch.from_numpy(np.array(object_labels)),
        action_boxes=torch.from_numpy(np.array(action_boxes).astype(np.float32)),
        action_labels=torch.from_numpy(np.array(action_labels)),
        image_id=item['file_name'],
        org_size=torch.as_tensor([int(img_hh), int(img_ww)]),
    )


def parse_one_gt_line(gt_line, scale=1):
    item = json.loads(gt_line)
    img_name = item['file_name']
    img_shape = item['height'], item['width']
    gt_boxes, ignored_boxes, total_boxes = get_det_annotation_from_odgt(item, img_shape, flip=0)
    interaction_boxes = get_hoi_annotation_from_odgt(item, total_boxes, scale)
    return dict(image_id=img_name, annotations=interaction_boxes)


def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "human_boxes" in target:
        boxes = target["human_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["human_boxes"] = boxes
    if "object_boxes" in target:
        boxes = target["object_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["object_boxes"] = boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["action_boxes"] = boxes
    return flipped_image, target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomAdjustImage(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.adjust_brightness(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        if random.random() < self.p:
            img = F.adjust_contrast(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        return img, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


def resize(image, target, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return oh, ow

    rescale_size = get_size_with_aspect_ratio(image_size=image.size, size=size, max_size=max_size)
    rescaled_image = F.resize(image, rescale_size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "human_boxes" in target:
        boxes = target["human_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["human_boxes"] = scaled_boxes
    if "object_boxes" in target:
        boxes = target["object_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["object_boxes"] = scaled_boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["action_boxes"] = scaled_boxes
    return rescaled_image, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


def crop(image, org_target, region):
    cropped_image = F.crop(image, *region)
    target = org_target.copy()
    i, j, h, w = region
    fields = ["human_labels", "object_labels", "action_labels"]

    if "human_boxes" in target:
        boxes = target["human_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["human_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("human_boxes")
    if "object_boxes" in target:
        boxes = target["object_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["object_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("object_boxes")
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["action_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("action_boxes")

    # remove elements for which the boxes or masks that have zero area
    if "human_boxes" in target and "object_boxes" in target:
        cropped_boxes = target['human_boxes'].reshape(-1, 2, 2)
        keep1 = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        cropped_boxes = target['object_boxes'].reshape(-1, 2, 2)
        keep2 = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        keep = keep1 * keep2
        if keep.any().sum() == 0:
            return image, org_target
        for field in fields:
            target[field] = target[field][keep]
    return cropped_image, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, (h, w))
        return crop(img, target, region)


class ToTensor(object):
    def __call__(self, img, target):
        return torchvision.transforms.functional.to_tensor(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = torchvision.transforms.functional.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "human_boxes" in target:
            boxes = target["human_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["human_boxes"] = boxes
        if "object_boxes" in target:
            boxes = target["object_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["object_boxes"] = boxes
        if "action_boxes" in target:
            boxes = target["action_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["action_boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def make_hico_transforms(image_set, test_scale=-1):
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            RandomAdjustImage(),
            RandomSelect(
                RandomResize(scales, max_size=1333),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(384, 600),
                    RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    if image_set == 'test':
        if test_scale == -1:
            return Compose([
                normalize,
            ])
        assert 400 <= test_scale <= 800, test_scale
        return Compose([
            RandomResize([test_scale], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


class HoiDetection(VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(HoiDetection, self).__init__(root, transforms, transform, target_transform)
        self.annotations = [parse_one_gt_line(l.strip()) for l in open(annFile, 'r').readlines()]
        self.transforms = transforms

    def __getitem__(self, index):
        ann = self.annotations[index]
        img_name = ann['image_id']
        target = ann['annotations']
        if 'train2014' in img_name:
            img_path = './data/vcoco/images/train2014/%s' % img_name
        elif 'val2014' in img_name:
            img_path = './data/vcoco/images/val2014/%s' % img_name
        else:
            raise NotImplementedError()
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)


def build(image_set, test_scale=-1):
    assert image_set in ['train', 'test'], image_set
    if image_set == 'train':
        annotation_file = './data/vcoco/vcoco_trainval_retag_hoitr.odgt'
    else:
        annotation_file = './data/vcoco/vcoco_test_retag_hoitr.odgt'
    dataset = HoiDetection(root='./data/vcoco', annFile=annotation_file,
                           transforms=make_hico_transforms(image_set, test_scale))
    return dataset
