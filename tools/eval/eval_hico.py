"""
    Evaluation for odgt format.
    Bohan Wang 2020-10-30
"""

import json
import numpy as np
import logging
from tqdm import tqdm


class hico():
    """
        output:
            object category id: coco[1, 90]
            verb category id: hico[1, 117]
    """
    def __init__(self, annotation_file):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.train_annotations = json.load(open(annotation_file.replace('test_hico.json', 'trainval_hico.json'), 'r'))
        self.overlap_iou = 0.5

        # add explation name for verb_name_dict
        self.verb_name_dict = []
        self.verb_name_dict_name = []

        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        self.file_name = []
        self.train_sum = {}

        # triple ids for 80 no_interaction and 520 interaction label
        self.no_inds = []
        self.in_inds = []

        # traverse test dataset to shouji gt
        # verb_name_dict: 600 lei represents hoi label
        # replace verb_name_dict with hoi label or sensibible name
        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            gt_bbox = gt_i['annotations']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                triplet = [gt_bbox[gt_hoi_i['subject_id']]['category_id'],
                           gt_bbox[gt_hoi_i['object_id']]['category_id'], gt_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    self.verb_name_dict.append(triplet)
                    # inverse_id always output the correspoinding idx in name
                    _verb_name = hico_action_name[hico_action_inverse_ids[triplet[2]]]
                    # add interaction/non_interaction qufen
                    if _verb_name == "no_interaction":
                        self.no_inds.append(self.verb_name_dict.index(triplet))
                    else:
                        self.in_inds.append(self.verb_name_dict.index(triplet))

                    _object_name = coco_object_name[coco_object_inverse_ids[triplet[1]]]
                    self.verb_name_dict_name.append(f"{_verb_name} {_object_name}")
                if self.verb_name_dict.index(triplet) not in self.sum_gt.keys():
                    self.sum_gt[self.verb_name_dict.index(triplet)] = 0
                self.sum_gt[self.verb_name_dict.index(triplet)] += 1
        assert len(self.no_inds) == 80, "number of no_interaction labels should be 80"
        assert len(self.in_inds) == 520, "number of interaction labels should be 520"

        # traverse trainval dataset to tongji number of instances
        for train_i in self.train_annotations:
            train_hoi = train_i['hoi_annotation']
            train_bbox = train_i['annotations']
            for train_hoi_i in train_hoi:
                if isinstance(train_hoi_i['category_id'], str):
                    train_hoi_i['category_id'] = int(train_hoi_i['category_id'].replace('\n', ''))
                triplet = [train_bbox[train_hoi_i['subject_id']]['category_id'],
                           train_bbox[train_hoi_i['object_id']]['category_id'], train_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    continue
                if self.verb_name_dict.index(triplet) not in self.train_sum.keys():
                    self.train_sum[self.verb_name_dict.index(triplet)] = 0
                self.train_sum[self.verb_name_dict.index(triplet)] += 1
        for i in range(len(self.verb_name_dict)):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
        self.r_inds = []
        self.c_inds = []
        for id in self.train_sum.keys():
            if self.train_sum[id] < 10:
                self.r_inds.append(id)
            else:
                self.c_inds.append(id)

        self.num_class = len(self.verb_name_dict)

    def evalution(self, predict_annot, save_mAP=None):
        for pred_i in predict_annot:
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            if len(gt_bbox) != 0:
                pred_bbox = self.add_One(pred_i['predictions'])  # convert zero-based to one-based indices
                if len(pred_bbox) == 0:  # To prevent compute_iou_mat
                    logging.warning(f"Image {pred_i['file_name']} pred NULL")
                    continue
                bbox_pairs, bbox_ov = self.compute_iou_mat(gt_bbox, pred_bbox)
                pred_hoi = pred_i['hoi_prediction']
                gt_hoi = gt_i['hoi_annotation']
                self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs, pred_bbox, bbox_ov)
            else:
                pred_bbox = self.add_One(pred_i['predictions'])  # convert zero-based to one-based indices
                for i, pred_hoi_i in enumerate(pred_i['hoi_prediction']):
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'],
                               pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                    verb_id = self.verb_name_dict.index(triplet)
                    self.tp[verb_id].append(0)
                    self.fp[verb_id].append(1)
                    self.score[verb_id].append(pred_hoi_i['score'])
        map = self.compute_map(save_mAP)
        return map

    def compute_map(self, save_mAP=None):
        logging.debug(f"total category = {self.num_class}")
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        name2ap = {}
        for i in range(len(self.verb_name_dict)):
            name = self.verb_name_dict_name[i]
            sum_gt = self.sum_gt[i]

            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i] = self.voc_ap(rec, prec)
            max_recall[i] = np.max(rec)
            logging.debug(f"class {self.verb_name_dict_name[i]} -- ap: {ap[i]} max recall:{max_recall[i]}")
            name2ap[name] = ap[i]

        mAP = np.mean(ap[:])
        mAP_rare = np.mean(ap[self.r_inds])
        mAP_nonrare = np.mean(ap[self.c_inds])
        mAP_inter = np.mean(ap[self.in_inds])
        mAP_noninter = np.mean(ap[self.no_inds])

        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print(
            f'mAP Full: {mAP}\nmAP rare: {mAP_rare}  mAP nonrare: {mAP_nonrare}\nmAP inter: {mAP_inter} mAP noninter: {mAP_noninter}\nmax recall: {m_rec}')
        print('--------------------')

        if save_mAP is not None:
            json.dump(name2ap, open(save_mAP, "w"))
        return mAP

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs, pred_bbox, bbox_ov):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i[
                    'object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_obj_ov = bbox_ov[pred_hoi_i['object_id']]
                    pred_sub_ov = bbox_ov[pred_hoi_i['subject_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    max_ov = 0
                    max_gt_id = 0
                    for gt_id in range(len(gt_hoi)):
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (
                                pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt = min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])],
                                            pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt > max_ov:
                                max_ov = min_ov_gt
                                max_gt_id = gt_id
                # logging.warning(f"pred_hoi_i{pred_hoi_i['category_id']} did in {list(self.fp.keys())[:5]}")
                # logging.warning(f"^^^pred_hoi_i = {pred_hoi_i}")
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'],
                           pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)
                if is_match == 1 and vis_tag[max_gt_id] == 0:
                    self.fp[verb_id].append(0)
                    self.tp[verb_id].append(1)
                    vis_tag[max_gt_id] = 1
                else:
                    self.fp[verb_id].append(1)
                    self.tp[verb_id].append(0)
                self.score[verb_id].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            # logging.warning(f" pred box is NULL")
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov = iou_mat.copy()
        iou_mat[iou_mat >= 0.5] = 1
        iou_mat[iou_mat < 0.5] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pairs_ov = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pairs_ov[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pairs_ov[pred_id].append(iou_mat_ov[match_pairs[0][i], pred_id])
        return match_pairs_dict, match_pairs_ov

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1)
            S_rec2 = (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line + 1) * (bottom_line - top_line + 1)
                return intersect / (sum_area - intersect)
        else:
            return 0

    def add_One(self, prediction):  # Add 1 to all coordinates
        for i, pred_bbox in enumerate(prediction):
            rec = pred_bbox['bbox']
            rec[0] += 1
            rec[1] += 1
            rec[2] += 1
            rec[3] += 1
        return prediction


# 80
coco_object_name = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

coco_object_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90]

# trans[1, 90] to [0, 80]
coco_object_inverse_ids = {idx: i for i, idx in enumerate(coco_object_valid_ids)}

# 117
hico_action_name = [
    'adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with',
    'buy', 'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook',
    'cut', 'cut_with', 'direct', 'drag', 'dribble', 'drink_with', 'drive',
    'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly',
    'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug',
    'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso', 'launch',
    'lick', 'lie_on', 'lift', 'light', 'load', 'lose', 'make', 'milk', 'move',
    'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay', 'peel',
    'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read',
    'release', 'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set',
    'shear', 'sign', 'sip', 'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze',
    'stab', 'stand_on', 'stand_under', 'stick', 'stir', 'stop_at', 'straddle',
    'swing', 'tag', 'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast', 'train',
    'turn', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip',
]

hico_name2id = {name: i + 1 for i, name in enumerate(hico_action_name)}

# trans[1, 117] to [0, 116]
hico_action_inverse_ids = {i: i - 1 for i in range(1, 118)}

# [1, 90]
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


def get_hoi_output(Image_dets, corre_mat=None):
    # 如果Object不满足 就要乘上0 纯评测
    output_hoi = []
    for Image_det in tqdm(Image_dets, desc="trans output into eval format"):
        Image_det = json.loads(Image_det)
        file_name = Image_det['image_id']
        output = {'predictions': [], 'hoi_prediction': [], 'file_name': file_name}
        count = 0
        for det in Image_det['hoi_list']:
            human_bbox = det['h_box']
            human_score = det['h_cls']

            object_bbox = det['o_box']
            object_score = det['o_cls']
            object_name = det["o_name"]
            object_cat = coco_classes_originID[object_name]

            inter_name = det["i_name"]
            inter_cat = hico_name2id[inter_name]
            inter_score = det["i_cls"]

            output['predictions'].append({'bbox': human_bbox, 'category_id': 1})
            human_idx = count
            count += 1
            output['predictions'].append({'bbox': object_bbox, 'category_id': object_cat})
            object_idx = count
            count += 1

            # inside denotes in training ocat_inside[0,80] icat_inside[0,116]
            ocat_inside = coco_object_inverse_ids[object_cat]
            icat_inside = hico_action_inverse_ids[inter_cat]

            final_score = corre_mat[icat_inside][ocat_inside] * human_score * object_score * inter_score

            output['hoi_prediction'].append({'subject_id': human_idx, 'object_id': object_idx, 'category_id': inter_cat, 'score': final_score})
        output_hoi.append(output)
    return output_hoi


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--eval_path", default="./data/hico/eval")

    args = parser.parse_args()

    # 1. transform hoi output
    with open(args.output_file, "r") as f:
        det = f.readlines()

    print(f"DEBUG: results = {len(det)}")
    corre_mat = np.load(os.path.join(args.eval_path, 'corre_hico.npy'))

    corre_mat = np.ones(shape=(117, 80))
    print(f"DEBUG: corre_mat shape = {corre_mat.shape}")
    output_hoi = get_hoi_output(det, corre_mat)

    # 2. evaluation
    hoi_eval = hico(os.path.join(args.eval_path, 'test_hico.json'))
    map = hoi_eval.evalution(output_hoi, save_mAP=None)
