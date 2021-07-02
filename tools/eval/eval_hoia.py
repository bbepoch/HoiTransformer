from tqdm import tqdm
import numpy as np
import json


coco_classes_originID = {
    "person": 1,
    "mobile_phone": 2,
    "cigarette": 3,
    "drink": 4,
    "food": 5,
    "bike": 6,
    "motorbike": 7,
    "horse": 8,
    "sports_ball": 9,
    "computer": 10,
    "document": 11,
}


hico_action_name = [
    'smoke', 'call', 'play(cellphone)', 'eat', 'drink', 'ride', 'hold', 'kick', 'read', 'play_computer'
]
hico_name2id = {name: i + 1 for i, name in enumerate(hico_action_name)}

coco_object_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

coco_object_inverse_ids = {idx: i for i, idx in enumerate(coco_object_valid_ids)}

hico_action_inverse_ids = {i: i - 1 for i in range(1, 11)}


class HOIAEval:
    def __init__(self, annotation_file):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.overlap_iou = 0.5
        self.verb_name_dict = {1: 'smoke', 2: 'call', 3: 'play(cellphone)', 4: 'eat', 5: 'drink',
                            6: 'ride', 7: 'hold', 8: 'kick', 9: 'read', 10: 'play (computer)'}
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        for i in list(self.verb_name_dict.keys()):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
            self.sum_gt[i] = 0
        self.file_name = []
        for gt_i in tqdm(self.annotations, desc='load gts: ', ncols=100):
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                if gt_hoi_i['category_id'] in list(self.verb_name_dict.keys()):
                    self.sum_gt[gt_hoi_i['category_id']] += 1
        self.num_class = len(list(self.verb_name_dict.keys()))
        print('prepare done')

    def evalution(self, prediction_json):
        predict_annot = json.load(open(prediction_json, 'r'))
        for pred_i in tqdm(predict_annot, desc='eval preds: ', ncols=100):
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            if 'predictions' in pred_i.keys():
                pred_bbox = pred_i['predictions']
            elif 'predcition' in pred_i.keys():
                pred_bbox = pred_i['prediction']
            elif 'annotations' in pred_i.keys():
                pred_bbox = pred_i['annotations']
            elif 'annotation' in pred_i.keys():
                pred_bbox = pred_i['annotation']
            else:
                print('prediction file keys error')
            if 'hoi_prediction' in pred_i.keys():
                pred_hoi = pred_i['hoi_prediction']
            elif 'hoi_predictions' in pred_i.keys():
                pred_hoi = pred_i['hoi_predictions']
            elif 'hoi_annotation' in pred_i.keys():
                pred_hoi = pred_i['hoi_annotation']
            else:
                print('prediction file keys error')
            gt_hoi = gt_i['hoi_annotation']
            bbox_pairs = self.compute_iou_mat(gt_bbox, pred_bbox)
            self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs)
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        for i in list(self.verb_name_dict.keys()):
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
            ap[i - 1] = self.voc_ap(rec,prec)
            max_recall[i-1] = np.max(rec)
            print('class {} --- ap: {}   max recall: {}'.format(
                i, ap[i-1], max_recall[i-1]))
        mAP = np.mean(ap[:])
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print('mAP: {}   max recall: {}'.format(mAP, m_rec))
        print('--------------------')
        return mAP

    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    for gt_id in np.nonzero(1 - vis_tag)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            vis_tag[gt_id] = 1
                            continue
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                if is_match == 1:
                    self.fp[pred_hoi_i['category_id']].append(0)
                    self.tp[pred_hoi_i['category_id']].append(1)

                else:
                    self.fp[pred_hoi_i['category_id']].append(1)
                    self.tp[pred_hoi_i['category_id']].append(0)
                self.score[pred_hoi_i['category_id']].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i
        iou_mat[iou_mat>= self.overlap_iou] = 1
        iou_mat[iou_mat< self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
        return match_pairs_dict

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

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
                intersect = (right_line - left_line) * (bottom_line - top_line)
                return intersect / (sum_area - intersect)
        else:
            return 0


def get_hoi_output(Image_dets, corre_mat=None):
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
            inter_cat = hico_name2id[str(inter_name)]
            inter_score = det["i_cls"]

            output['predictions'].append({'bbox': human_bbox, 'category_id': 1})
            human_idx = count
            count += 1
            output['predictions'].append({'bbox': object_bbox, 'category_id': object_cat})
            object_idx = count
            count += 1

            # inside denotes in training ocat_inside[0,80] icat_inside[0,116]
            ocat_inside = coco_object_inverse_ids[int(object_cat)]
            icat_inside = hico_action_inverse_ids[int(inter_cat)]

            final_score = corre_mat[icat_inside][ocat_inside] * human_score * object_score * inter_score

            output['hoi_prediction'].append({'subject_id': human_idx, 'object_id': object_idx, 'category_id': inter_cat, 'score': final_score})
        output_hoi.append(output)
    return output_hoi


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--eval_path", default="./data/hoia/eval")
    args = parser.parse_args()

    with open(args.output_file, "r") as f:
        det = f.readlines()
    print(f"DEBUG: results = {len(det)}")

    corre_mat = np.load(os.path.join(args.eval_path, 'corre_hoia.npy'))
    print(f"DEBUG: corre_mat shape = {corre_mat.shape}")

    output_hoi = get_hoi_output(det, corre_mat)
    new_out_file = args.output_file.replace('.json', '_hoia.json')
    with open(new_out_file, 'w') as f:
        json.dump(output_hoi, f)

    annotation_file = os.path.join(args.eval_path, 'test_hoia.json')
    eval_demo = HOIAEval(annotation_file)
    eval_demo.evalution(new_out_file)
