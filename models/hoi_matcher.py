# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["action_pred_logits"].shape[:2]  # 2, 100

        # We flatten to compute the cost matrices in a batch
        human_out_prob = outputs["human_pred_logits"].flatten(0, 1).softmax(-1)  # [bs * num_queries, num_classes]
        human_out_bbox = outputs["human_pred_boxes"].flatten(0, 1)  # [bs * num_queries, 4]
        object_out_prob = outputs["object_pred_logits"].flatten(0, 1).softmax(-1)  # [bs * num_queries, num_classes]
        object_out_bbox = outputs["object_pred_boxes"].flatten(0, 1)  # [bs * num_queries, 4]
        action_out_prob = outputs["action_pred_logits"].flatten(0, 1).softmax(-1)  # [bs * num_queries, num_classes]

        # Also concat the target labels and boxes
        human_tgt_ids = torch.cat([v["human_labels"] for v in targets])
        human_tgt_box = torch.cat([v["human_boxes"] for v in targets])
        object_tgt_ids = torch.cat([v["object_labels"] for v in targets])
        object_tgt_box = torch.cat([v["object_boxes"] for v in targets])
        action_tgt_ids = torch.cat([v["action_labels"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        human_cost_class = -human_out_prob[:, human_tgt_ids]
        object_cost_class = -object_out_prob[:, object_tgt_ids]
        action_cost_class = -action_out_prob[:, action_tgt_ids]

        # Compute the L1 cost between boxes
        human_cost_bbox = torch.cdist(human_out_bbox, human_tgt_box, p=1)
        object_cost_bbox = torch.cdist(object_out_bbox, object_tgt_box, p=1)

        # Compute the giou cost betwen boxes
        human_cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(human_out_bbox), box_cxcywh_to_xyxy(human_tgt_box))
        object_cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(object_out_bbox), box_cxcywh_to_xyxy(object_tgt_box))

        beta_1, beta_2 = 1.2, 1
        alpha_h, alpha_o, alpha_r = 1, 1, 2
        l_cls_h = alpha_h * self.cost_class * human_cost_class
        l_cls_o = alpha_o * self.cost_class * object_cost_class
        l_cls_r = alpha_r * self.cost_class * action_cost_class
        l_box_h = self.cost_bbox * human_cost_bbox + self.cost_giou * human_cost_giou
        l_box_o = self.cost_bbox * object_cost_bbox + self.cost_giou * object_cost_giou
        l_cls_all = (l_cls_h + l_cls_o + l_cls_r) / (alpha_h + alpha_o + alpha_r)
        l_box_all = (l_box_h + l_box_o) / 2
        C = beta_1 * l_cls_all + beta_2 * l_box_all

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["human_boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        result = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return result


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
