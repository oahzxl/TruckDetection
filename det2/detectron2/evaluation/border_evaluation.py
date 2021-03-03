# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import cv2
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator

# category_id = {'truck': 0}
category_id = {'legal': 0,'illegal':0,'empty':0}
class BorderEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, crop=False, task=None, expand_ratio=1):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._crop = crop
        self._task = task
        self._expand_ratio = expand_ratio


        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self.scores_threshold = 0.5#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.iou_threshold = 0.4#cfg.MODEL.ROI_HEADS.IOU_THRESH_TEST

        self.save_num = 564

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            prediction["file_name"] = input["file_name"]
            prediction["height"] = input["height"]
            prediction["width"] = input["width"]

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "proposals" in self._predictions[0]:
            self._eval_box_proposals()
        if "instances" in self._predictions[0]:
            self._eval_predictions(set(self._tasks))
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        class_list = self._metadata.get("thing_classes")
        res = []
        for c_i, c in enumerate(class_list):
            result_dict = {}
            result_dict["class"] = class_list[c_i]
            result_dict["Recall"]  = 0
            result_dict["Precision"]  = 0
            result_dict["GtNums"]  = 0
            result_dict["PNums"]  = 0
            result_dict["TP"]  = 0
            result_dict["FP"]  = 0
        #  result_dict["FN"]  = 0
            res.append(result_dict)

        self._logger.info("Preparing results for data format ...")
        for pred in self._predictions:
            img_path = pred["file_name"]
            json_path = img_path.replace("jpg", "json")
            if not os.path.exists(json_path):
                print("not find ", json_path)

            with open(json_path,'r') as load_f:
                gt_json = json.load(load_f)
            ori_image = cv2.imread(img_path)
            pred_image = copy.deepcopy(ori_image)
            proposal_image = copy.deepcopy(ori_image)
            pred_label = pred["instances"].pred_classes.numpy()
            pred_box = pred["instances"].pred_boxes.tensor.numpy()
            pred_score = pred["instances"].scores.numpy()
            keep_index = (pred_score > self.scores_threshold)
            pred_score = pred_score[keep_index]
            pred_label = pred_label[keep_index]
            pred_box = pred_box[keep_index]
            objects = gt_json["shapes"]
            save_num = 1

            if len(objects) < 1:
                print(len(objects))
                continue

            have_hitted=[]
            gt_index = 0
            for obj in objects:
                gt_label = int(category_id[obj["label"]])
                if gt_label != 0 and gt_label != 1 and gt_label != 2:
                    gt_index += 1
                    continue
                x1 = obj["points"][0][0]
                y1 = obj["points"][0][1]
                x2 = obj["points"][1][0]
                y2 = obj["points"][1][1]
                x_min = np.min((x1,x2))
                x_max = np.max((x1,x2))
                y_min = np.min((y1,y2))
                y_max = np.max((y1,y2))
                res[gt_label]["GtNums"] += 1
                ori_image = cv2.rectangle(ori_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 255, 0), 2)
                match_index = -1
                max_iou = 0
                match_box =[]
                for i in range(pred_label.shape[0]):
                    x1, y1, x2, y2 = pred_box[i]
                    proposal_image = cv2.putText(proposal_image,
                                        "rpn:" + str(gt_label) + " "+ str(pred_score[i])[:3],
                                        (x1, y1), font, 1, (255, 0, 0),
                                        1)
                    proposal_image = cv2.rectangle(proposal_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    # if (pred_label[i] == 1 & gt_label == 2) or (pred_label[i] == 0 & (gt_label == 0 or gt_label == 1)) and (i not in have_hitted):
                    if i not in have_hitted:
                    # if pred_label[i] == gt_label and (i not in have_hitted):
                        iou = compute_iou([x_min, y_min, x_max, y_max], pred_box[i])
                        if iou > self.iou_threshold:
                            if iou > max_iou:
                                max_iou = iou
                                match_index = i

                if match_index != -1:
                    res[gt_label]["TP"] += 1
                    have_hitted.append(match_index)
                    x1, y1, x2, y2 = pred_box[match_index]

                # crop images
                if self._crop:
                    try:
                        expand_ratio = self._expand_ratio
                        y_min2 = y_min * (expand_ratio / 2 + 0.5) - (expand_ratio / 2 - 0.5) * y_max
                        y_max2 = y_max * (expand_ratio / 2 + 0.5) - (expand_ratio / 2 - 0.5) * y_min
                        x_min2 = x_min * (expand_ratio / 2 + 0.5) - (expand_ratio / 2 - 0.5) * x_max
                        x_max2 = x_max * (expand_ratio / 2 + 0.5) - (expand_ratio / 2 - 0.5) * x_min
                        y_min2 = y_min2 if y_min2 > 0 else 0
                        y_max2 = y_max2 if y_max2 < pred_image.shape[0] else pred_image.shape[0]
                        x_min2 = x_min2 if x_min2 > 0 else 0
                        x_max2 = x_max2 if x_max2 < pred_image.shape[1] else pred_image.shape[1]
                        # path = '/home/jhy/data/gt1.2/train/' + obj["label"] + '/' + str(self.save_num) + '.jpg'
                        save_name = img_path.split("/")[-1].split(".jpg")[0]

                        path = '/home/zxl/Truck-Detection/data/crop/' + self._task + '/' \
                               + obj["label"] + '/' + save_name + "_" + str(save_num) + '.jpg'

                        print(path)
                        cv2.imwrite(path, pred_image[int(y_min2):int(y_max2), int(x_min2):int(x_max2)])
                        save_num += 1
                    except:
                        import pdb; pdb.set_trace()
                        pred_image = cv2.putText(pred_image,
                                            "tp:" + str(gt_label) + " "+ str(pred_score[match_index])[:3],
                                            (x1, y1), font, 1, (255, 0, 0),
                                            1)
                        pred_image = cv2.rectangle(pred_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            for j in range(pred_box.shape[0]):
                l_index = pred_label[j]
                x1, y1, x2, y2 = pred_box[j]
                if j not in have_hitted:
                    res[l_index]["FP"] += 1
                    # pred_image = cv2.putText(pred_image,
                    #                     "fp:" + str(l_index) +" "+ str(pred_score[j])[:3],
                    #                     (x2, y2), font, 1, (0, 0, 255),
                    #                     1)
                    # pred_image = cv2.rectangle(pred_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                res[l_index]["PNums"] += 1
            # cv2.imwrite('/home/zxl/Truck-Detection/data/crop/vis/' + os.path.basename(img_path), pred_image)
        for r in res:
            if r["GtNums"] == 0 or r["PNums"] == 0:
                print("=========== r['GtNums', 'PNums'] =========", r["GtNums"], r["PNums"])
                r["Recall"] = -1
                r["Precision"] = -1

            else:
                r["Recall"] = r["TP"] / r["GtNums"]
                r["Precision"] = r["TP"] / r["PNums"]
            print(r)

        return None


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
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
        return (intersect / (sum_area - intersect))*1.0
