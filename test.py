# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------

import argparse
import json
import random
import os
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from datasets.hico import hoi_interaction_names as hoi_interaction_names_hico
from datasets.hico import coco_instance_ID_to_name as coco_instance_ID_to_name_hico
from datasets.hoia import hoi_interaction_names as hoi_interaction_names_hoia
from datasets.hoia import coco_instance_ID_to_name as coco_instance_ID_to_name_hoia
from datasets.vcoco import hoi_interaction_names as hoi_interaction_names_vcoco
from datasets.vcoco import coco_instance_ID_to_name as coco_instance_ID_to_name_vcoco
from models import build_model
import util.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Backbone.
    parser.add_argument('--backbone', choices=['resnet50', 'resnet101'], required=True,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # Transformer.
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss.
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher.
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # Loss coefficients.
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # Dataset parameters.
    parser.add_argument('--dataset_file', choices=['hico', 'vcoco', 'hoia'], required=True)

    parser.add_argument('--model_path', required=True,
                        help='Path of the model to evaluate.')
    parser.add_argument('--log_dir', default='./',
                        help='path where to save temporary files in test')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # Distributed training parameters.
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Visualization.
    parser.add_argument('--max_to_viz', default=10, type=int, help='number of images to visualize')
    parser.add_argument('--save_image', action='store_true', help='whether to save visualization images')
    return parser


def random_color():
    rdn = random.randint(1, 1000)
    b = int(rdn * 997) % 255
    g = int(rdn * 4447) % 255
    r = int(rdn * 6563) % 255
    return b, g, r


def intersection(box_a, box_b):
    # box: x1, y1, x2, y2
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    return float((x2 - x1 + 1) * (y2 - y1 + 1))


def IoU(box_a, box_b):
    inter = intersection(box_a, box_b)
    box_a_area = (box_a[2]-box_a[0]+1) * (box_a[3]-box_a[1]+1)
    box_b_area = (box_b[2]-box_b[0]+1) * (box_b[3]-box_b[1]+1)
    union = box_a_area + box_b_area - inter
    return inter / float(max(union, 1))


def triplet_nms(hoi_list):
    hoi_list.sort(key=lambda x: x['h_cls'] * x['o_cls'] * x['i_cls'], reverse=True)
    mask = [True] * len(hoi_list)
    for idx_x in range(len(hoi_list)):
        if mask[idx_x] is False:
            continue
        for idx_y in range(idx_x+1, len(hoi_list)):
            x = hoi_list[idx_x]
            y = hoi_list[idx_y]
            iou_human = IoU(x['h_box'], y['h_box'])
            iou_object = IoU(x['o_box'], y['o_box'])
            if iou_human > 0.5 and iou_object > 0.5 and x['i_name'] == y['i_name'] and x['o_name'] == y['o_name']:
                mask[idx_y] = False
    new_hoi_list = []
    for idx in range(len(mask)):
        if mask[idx] is True:
            new_hoi_list.append(hoi_list[idx])
    return new_hoi_list


def inference_on_data(args, model_path, image_set, max_to_viz=10, test_scale=-1):
    assert image_set in ['train', 'test'], image_set
    checkpoint = torch.load(model_path, map_location='cpu')
    epoch = checkpoint['epoch']
    print('epoch:', epoch)

    device = torch.device(args.device)
    model, criterion = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    dataset_val = build_dataset(image_set=image_set, args=args, test_scale=test_scale)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    log_dir = os.path.join(args.log_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    assert os.path.exists(log_dir), log_dir
    file_name = 'result_s%03d_e%03d_%s_%s.pkl' \
                % (0 if test_scale == -1 else test_scale, epoch, args.dataset_file, args.backbone)
    file_path = os.path.join(log_dir, file_name)
    if os.path.exists(file_path):
        print('step1: file exists, inference done.')
        return file_path

    p_bar = tqdm(total=max_to_viz)

    idx_batch, result_list = 0, []
    for samples, targets in data_loader_val:
        idx_batch += 1
        if idx_batch >= max_to_viz:
            break
        id_list = [targets[idx]['image_id'] for idx in range(len(targets))]
        org_sizes = [targets[idx]['org_size'] for idx in range(len(targets))]
        samples = samples.to(device)
        outputs = model(samples)
        action_pred_logits = outputs['action_pred_logits']
        object_pred_logits = outputs['object_pred_logits']
        object_pred_boxes = outputs['object_pred_boxes']
        human_pred_logits = outputs['human_pred_logits']
        human_pred_boxes = outputs['human_pred_boxes']
        result_list.append(dict(
            id_list=id_list,
            org_sizes=org_sizes,
            action_pred_logits=action_pred_logits.detach().cpu(),
            object_pred_logits=object_pred_logits.detach().cpu(),
            object_pred_boxes=object_pred_boxes.detach().cpu(),
            human_pred_logits=human_pred_logits.detach().cpu(),
            human_pred_boxes=human_pred_boxes.detach().cpu(),
        ))
        p_bar.update()

    with open(file_path, 'wb') as f:
        torch.save(result_list, f)
    print('step1: inference done.')
    return file_path


def parse_model_result(args, result_path, hoi_th=0.9, human_th=0.5, object_th=0.8, max_to_viz=10):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file
    if args.dataset_file == 'hico':
        num_classes = 91
        num_actions = 118
        top_k = 200
        hoi_interaction_names = hoi_interaction_names_hico
        coco_instance_id_to_name = coco_instance_ID_to_name_hico
    elif args.dataset_file == 'vcoco':
        num_classes = 91
        num_actions = 30
        top_k = 35
        hoi_interaction_names = hoi_interaction_names_vcoco
        coco_instance_id_to_name = coco_instance_ID_to_name_vcoco
    else:
        num_classes = 12
        num_actions = 11
        top_k = 35
        hoi_interaction_names = hoi_interaction_names_hoia
        coco_instance_id_to_name = coco_instance_ID_to_name_hoia

    with open(result_path, 'rb') as f:
        output_list = torch.load(f, map_location='cpu')

    idx_batch, final_hoi_result_list = 0, []
    for outputs in tqdm(output_list):  # batch level
        idx_batch += 1
        if idx_batch >= max_to_viz:
            break
        img_id_list = outputs['id_list']
        org_sizes = outputs['org_sizes']
        action_pred_logits = outputs['action_pred_logits']
        object_pred_logits = outputs['object_pred_logits']
        object_pred_boxes = outputs['object_pred_boxes']
        human_pred_logits = outputs['human_pred_logits']
        human_pred_boxes = outputs['human_pred_boxes']
        assert len(action_pred_logits) == len(img_id_list)

        for idx_img in range(len(action_pred_logits)):
            image_id = img_id_list[idx_img]
            hh, ww = org_sizes[idx_img]

            act_cls = torch.nn.Softmax(dim=1)(action_pred_logits[idx_img]).detach().cpu().numpy()[:, :-1]
            human_cls = torch.nn.Softmax(dim=1)(human_pred_logits[idx_img]).detach().cpu().numpy()[:, :-1]
            object_cls = torch.nn.Softmax(dim=1)(object_pred_logits[idx_img]).detach().cpu().numpy()[:, :-1]
            human_box = human_pred_boxes[idx_img].detach().cpu().numpy()
            object_box = object_pred_boxes[idx_img].detach().cpu().numpy()

            keep = (act_cls.argmax(axis=1) != num_actions)
            keep = keep * (human_cls.argmax(axis=1) != 2)
            keep = keep * (object_cls.argmax(axis=1) != num_classes)
            keep = keep * (act_cls > hoi_th).any(axis=1)
            keep = keep * (human_cls > human_th).any(axis=1)
            keep = keep * (object_cls > object_th).any(axis=1)

            human_idx_max_list = human_cls[keep].argmax(axis=1)
            human_val_max_list = human_cls[keep].max(axis=1)
            human_box_max_list = human_box[keep]
            object_idx_max_list = object_cls[keep].argmax(axis=1)
            object_val_max_list = object_cls[keep].max(axis=1)
            object_box_max_list = object_box[keep]
            keep_act_scores = act_cls[keep]

            keep_act_scores_1d = keep_act_scores.reshape(-1)
            top_k_idx_1d = np.argsort(-keep_act_scores_1d)[:top_k]
            box_action_pairs = [(idx_1d // num_actions, idx_1d % num_actions) for idx_1d in top_k_idx_1d]

            hoi_list = []
            for idx_box, idx_action in box_action_pairs:
                # action
                i_box = (0, 0, 0, 0)
                i_cls = keep_act_scores[idx_box, idx_action]
                i_name = hoi_interaction_names[int(idx_action)]
                if i_name in ['__background__']:
                    continue
                # human
                cid = human_idx_max_list[idx_box]
                cx, cy, w, h = human_box_max_list[idx_box]
                cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
                h_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
                h_cls = human_val_max_list[idx_box]
                h_name = coco_instance_id_to_name[int(cid)]
                # object
                cid = object_idx_max_list[idx_box]
                cx, cy, w, h = object_box_max_list[idx_box]
                cx, cy, w, h = cx * ww, cy * hh, w * ww, h * hh
                o_box = list(map(int, [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]))
                o_cls = object_val_max_list[idx_box]
                o_name = coco_instance_id_to_name[int(cid)]
                if i_cls < hoi_th or h_cls < human_th or o_cls < object_th:
                    continue
                pp = dict(
                    h_box=h_box, o_box=o_box, i_box=i_box, h_cls=float(h_cls), o_cls=float(o_cls),
                    i_cls=float(i_cls), h_name=h_name, o_name=o_name, i_name=i_name,
                )
                hoi_list.append(pp)

            hoi_list = triplet_nms(hoi_list)
            item = dict(image_id=image_id, hoi_list=hoi_list)
            final_hoi_result_list.append(item)
    return final_hoi_result_list


def draw_on_image(args, image_id, hoi_list, image_path):
    img_name = image_id
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file
    if args.dataset_file == 'hico':
        if 'train2015' in img_name:
            img_path = './data/hico/images/train2015/%s' % img_name
        elif 'test2015' in img_name:
            img_path = './data/hico/images/test2015/%s' % img_name
        else:
            raise NotImplementedError()
    elif args.dataset_file == 'vcoco':
        if 'train2014' in img_name:
            img_path = './data/vcoco/images/train2014/%s' % img_name
        elif 'val2014' in img_name:
            img_path = './data/vcoco/images/val2014/%s' % img_name
        else:
            raise NotImplementedError()
    else:
        if 'trainval' in img_name:
            img_path = './data/hoia/images/trainval/%s' % img_name
        elif 'test' in img_name:
            img_path = './data/hoia/images/test/%s' % img_name
        else:
            raise NotImplementedError()

    img_result = cv2.imread(img_path, cv2.IMREAD_COLOR)
    for idx_box, hoi in enumerate(hoi_list):
        color = random_color()
        # action
        i_cls, i_name = hoi['i_cls'], hoi['i_name']
        cv2.putText(img_result, '%s:%.4f' % (i_name, i_cls),
                    (10, 50 * idx_box + 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        # human
        x1, y1, x2, y2 = hoi['h_box']
        h_cls, h_name = hoi['h_cls'], hoi['h_name']
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_result, '%s:%.4f' % (h_name, h_cls), (x1, y2), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        # object
        x1, y1, x2, y2 = hoi['o_box']
        o_cls, o_name = hoi['o_cls'], hoi['o_name']
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_result, '%s:%.4f' % (o_name, o_cls), (x1, y2), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    if img_result.shape[0] > 640:
        ratio = img_result.shape[0] / 640
        img_result = cv2.resize(img_result, (int(img_result.shape[1] / ratio), int(img_result.shape[0] / ratio)))
    cv2.imwrite(image_path, img_result)


def eval_once(args, model_result_path, hoi_th=0.9, human_th=0.5, object_th=0.8, max_to_viz=10, save_image=False):
    assert args.dataset_file in ['hico', 'vcoco', 'hoia'], args.dataset_file

    hoi_result_list = parse_model_result(
        args=args,
        result_path=model_result_path,
        hoi_th=hoi_th,
        human_th=human_th,
        object_th=object_th,
        max_to_viz=max_to_viz,
    )

    result_file = model_result_path.replace('.pkl', '.json')
    with open(result_file, 'w') as writer:
        for idx_img, item in enumerate(hoi_result_list):
            writer.write(json.dumps(item) + '\n')
            if save_image and idx_img < max_to_viz:
                img_path = '%s/dt_%02d.jpg' % (os.path.dirname(model_result_path), idx_img)
                draw_on_image(args, item['image_id'], item['hoi_list'], image_path=img_path)

    os.system('echo %s >> final_report.txt' % result_file)
    if args.dataset_file == 'hico':
        os.system('python3 tools/eval/eval_hico.py --output_file=%s >> final_report.txt' % result_file)
    elif args.dataset_file == 'vcoco':
        os.system('python3 tools/eval/eval_vcoco.py --output_file=%s >> final_report.txt' % result_file)
    else:
        os.system('python3 tools/eval/eval_hoia.py --output_file=%s >> final_report.txt' % result_file)
    os.system('echo %s >> final_report.txt' % '%f %f %f\n' % (human_th, object_th, hoi_th))
    print(human_th, object_th, hoi_th, '--------------------above')


def run_and_eval(args, model_path, test_scale, max_to_viz=10, save_image=False):
    model_output_file = inference_on_data(
        args=args,
        model_path=model_path,
        image_set='test',
        test_scale=test_scale,
        max_to_viz=max_to_viz,
    )

    for human_th in [0.0]:
        for object_th in [0.0]:
            for hoi_th in [0.0]:
                eval_once(
                    args=args,
                    model_result_path=model_output_file,
                    hoi_th=hoi_th,
                    human_th=human_th,
                    object_th=object_th,
                    max_to_viz=max_to_viz,
                    save_image=save_image,
                )
    pass


def main():
    """
    python3 test.py --dataset_file=hico --backbone=resnet50 --batch_size=1 --log_dir=./ --model_path=your_model_path
    """
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)

    scales = [672]
    for test_scale in scales:
        for model_path in [
            args.model_path,
        ]:
            os.system('echo %s >> final_report.txt' % model_path)
            print(model_path)
            run_and_eval(
                args=args,
                model_path=model_path,
                test_scale=test_scale,
                max_to_viz=args.max_to_viz if args.save_image else 200*100,
                save_image=args.save_image,
            )
    print('done')


if __name__ == '__main__':
    main()
