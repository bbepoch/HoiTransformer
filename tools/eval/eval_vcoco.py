from vsrl_eval import VCOCOeval
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
import pickle

# -------------------actions predfine ----------------------------------
ACTION_NAMES = {'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                'kick', 'point', 'read', 'snowboard'}  # 4
ACTION_AGENT = {f"{action}_agent" for action in ACTION_NAMES}

ICAN_ACTION_ROLE = {"cut_instr": 2, "snowboard_instr": 21, "cut_obj": 4, "surf_instr": 0,
                     "skateboard_instr": 26, "kick_obj": 7, "eat_obj": 9, "carry_obj": 14,
                     "throw_obj": 15, "eat_instr": 16, "smile": 17, "look_obj": 18, "hit_instr": 19,
                     "hit_obj": 20, "ski_instr": 1, "run": 22, "sit_instr": 10, "read_obj": 24,
                     "ride_instr": 5, "walk": 3, "point_instr": 23, "jump_instr": 11,
                     "work_on_computer_instr": 8, "hold_obj": 25, "drink_instr": 13, "lay_instr": 12,
                     "talk_on_phone_instr": 6, "stand": 27, "catch_obj": 28}
INTRANSITIVE_VERBS = {"smile", "run", "walk", "stand"}

ACTION_ID2ROLE = {v: k for k, v in ICAN_ACTION_ROLE.items()}

ACTION_ROLE = {'hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk',  # 5
                            'look_obj', 'hit_instr', 'eat_instr', 'jump_instr', 'lay_instr',  # 5
                            'talk_on_phone_instr', 'carry_obj', 'throw_obj', 'catch_obj',  # 4
                            'cut_instr', 'run', 'work_on_computer_instr', 'ski_instr',  # 4
                            'surf_instr', 'skateboard_instr', 'smile', 'drink_instr',  # 4
                            'kick_obj', 'point_instr', 'read_obj', 'snowboard_instr',   # 4
                            'hit_obj', 'cut_obj', 'eat_obj'} # add 3


# 原因是因为在训练集里只有25种verb，但是有标签的有29种[0-28]
valid_ids_verb = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28]
# [0, 28] -> [0, 24]
vcoco_action_invrese_ids = {k: i for i, k in enumerate(valid_ids_verb)}

coco_object_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90]
# trans[1, 90] to [0, 79]
coco_object_inverse_ids = {idx: i for i, idx in enumerate(coco_object_valid_ids)}

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

# ---------------------------------------


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
            inter_cat = ICAN_ACTION_ROLE[inter_name]
            inter_score = det["i_cls"]

            output['predictions'].append({'bbox': human_bbox, 'category_id': 1})
            human_idx = count
            count += 1
            output['predictions'].append({'bbox': object_bbox, 'category_id': object_cat})
            object_idx = count
            count += 1

            # inside denotes in training ocat_inside[0,80] icat_inside[0,24]
            ocat_inside = coco_object_inverse_ids[object_cat]
            icat_inside = vcoco_action_invrese_ids[inter_cat]

            final_score = corre_mat[icat_inside][ocat_inside] * inter_score * object_score * human_score

            output['hoi_prediction'].append({'subject_id': human_idx, 'object_id': object_idx, 'category_id': inter_cat, 'score': final_score})
        output_hoi.append(output)
    return output_hoi


def post_process(output, fname2cocoid):
  # --------- map filename to img_ids for evlauation ---------
  with open(fname2cocoid, "r") as f:
    name2id = json.load(f)

  detections = []
  for oneimage_output in tqdm(output, desc="trans into vcoco format"):

    # -----------Human Center Format --------------
    fname = oneimage_output["file_name"]
    image_id = name2id[fname]
    This_image = defaultdict(dict)
    obj_proposals = oneimage_output['predictions']
    for pred_hoi in oneimage_output["hoi_prediction"]:
      action_role = ACTION_ID2ROLE[pred_hoi['category_id']]
      action_object_score = pred_hoi['score']
      human_index = pred_hoi['subject_id']
      # role AP
      if action_role in INTRANSITIVE_VERBS:
        # walk smile run stand
        print("LOCATE A BUG.")
        pass
      else:
        object_index = pred_hoi['object_id']
        object_box = obj_proposals[object_index]
        This_image[human_index][action_role] = np.array(object_box["bbox"] + [action_object_score])

    # ---------- 补充多余的空结果 -----------
    for human_index, human_dict in This_image.items():
      human_dict["person_box"] = obj_proposals[human_index]["bbox"]
      human_dict["image_id"] = image_id
      # add those impossible actions
      for role in ACTION_ROLE:
        if human_dict.get(role) is None:
          human_dict[role] = np.append(np.full(4, np.nan).reshape(1, 4), 0)
      for agent in ACTION_AGENT:
        if human_dict.get(agent) is None:
          human_dict[agent] = 0
      detections.append(human_dict)

  return detections


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--eval_path", default="./data/vcoco/eval")
    parser.add_argument("--use_prior", action="store_true")

    args = parser.parse_args()

    # ---------------- vcoco evaluation --------------------
    vsrl_annot_file = f"{args.eval_path}/vcoco_test.json"
    coco_file = f"{args.eval_path}/instances_vcoco_all_2014.json"
    split_file = f"{args.eval_path}/vcoco_test.ids"
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    fname2cocoid = f"{args.eval_path}/fname2imgid_test.json"
    output_file = f"{args.eval_path}/test_vcoco_final.pkl"

    # 1. transform hoi output
    with open(args.output_file, "r") as f:
        det = f.readlines()
    print(f"DEBUG: results = {len(det)}")

    if args.use_prior:
        corre_mat = np.load(os.path.join(args.eval_path, 'corre_verbcoco_v2.npy'))
    else:
        corre_mat = np.ones(shape=(25, 80))
    print(f"DEBUG: corre_mat shape = {corre_mat.shape}")

    output_hoi = get_hoi_output(det, corre_mat)

    trans_output_hoi = post_process(output_hoi, fname2cocoid)
    pickle.dump(trans_output_hoi, open(output_file, "wb"))
    vcocoeval._do_eval(output_file, ovr_thresh=0.5)
