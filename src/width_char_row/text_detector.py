import logging
import os
import warnings
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
from doctr.models import detection_predictor
from doctr.models.detection.predictor import DetectionPredictor
from scipy.signal import savgol_filter, find_peaks

from width_char_row.bbox import BBox
from width_char_row.binarization import binarize # наша бинаризация


'''

В этом классе код функций сегментации строк:
sort_bboxes_by_coordinates(..) - сегментация чисто по координатам боксов слов (BBox)
sort_bboxes(..)  - сегментация с использованием изображения

Если понадобятся получить ббоксы текстовые, то модель детекции текста - это doctr (https://github.com/mindee/doctr)


'''


class TextDetector:

    def __init__(self, on_gpu: bool, checkpoint_path: Optional[str], with_vertical_text_detection: bool = False,
                 *, config: dict) -> None:
        self.logger = config.get("logger", logging.getLogger())
        self._set_device(on_gpu)
        self._net = None
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.config = config
        self.with_vertical_text_detection = with_vertical_text_detection

    def _set_device(self, on_gpu: bool) -> None:
        """
        Set device configuration
        """
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.location = lambda storage, loc: storage.cuda()
        else:
            self.device = torch.device("cpu")
            self.location = 'cpu'

    @property
    def net(self) -> DetectionPredictor:
        """
        Predict consists from List of bounding box with a format ( left, top, right, bottom, confidence )
        :return: Text Localization model
        """

        # lazy loading net
        if not self._net:
            '''
            Note: - model db_resnet50 with weights = "db_resnet50-ac60cadc.pt" are more accuracy
            (can detect numder of certificate);
                     - model db_resnet50_rotation with weights = "db_resnet50_rotation-1138863a.pt"
            (can detect text with different orientation, but can not detect some text.
            TODO: when we introduce image classifier, we need to call db_resnet50_rotation for scene images, and
                  db_resnet50 for different documents with background (example certificates)
            '''
            if self.with_vertical_text_detection:
                self._net = detection_predictor(arch='db_resnet50_rotation', pretrained=False).eval().to(self.device)
                self._load_weights(self._net, "db_resnet50_rotation-1138863a.pt")
            else:
                self._net = detection_predictor(arch='db_resnet50', pretrained=False).eval().to(self.device)
                self._load_weights(self._net, "db_resnet50-ac60cadc.pt")

        return self._net

    def _load_weights(self, net: DetectionPredictor, name_weight: str) -> None:
        path_checkpoint = os.path.join(self.checkpoint_path, name_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.model.load_state_dict(torch.load(path_checkpoint, map_location=self.location))
            self.logger.info('Weights were loaded from {}'.format(path_checkpoint))

    def predict(self, image: np.ndarray) -> Tuple[List[BBox], List[float]]:
        """
        :param image: input batch of image with some text
        :return: prediction: List[BBox] - text coords prediction, List[float] - confidence of prediction
        """
        h, w = image.shape[:2]
        boxes, confs = [], []
        batch_preds = self.net([image])

        # for pred in batch_preds[0]:
        for pred in batch_preds[0]["words"]:
            left = int(pred[0] * w)
            top = int(pred[1] * h)
            bottom = int(pred[3] * h)
            right = int(pred[2] * w)
            box = BBox.from_two_points(top_left=(left, top),
                                       bottom_right=(right, bottom))
            boxes.append(box)
            confs.append(pred[4])

        return boxes, confs

    def _is_on_same_line(self, box_one: np.ndarray, box_two: np.ndarray, intersection_thr: float) -> float:
        box_one_start_y = box_one[0, 1]
        box_one_end_y = box_one[1, 1]
        box_two_start_y = box_two[0, 1]
        box_two_end_y = box_two[1, 1]

        inter = max(min(box_one_end_y, box_two_end_y) - max(box_one_start_y, box_two_start_y), 0)
        if inter == 0:
            return False

        union = min(box_one_end_y - box_one_start_y, box_two_end_y - box_two_start_y)

        return inter / union > intersection_thr

    def _segment_lines(self, box_group: np.ndarray, intersection_thr: float) -> Tuple[np.ndarray, List]:
        box_group = box_group[np.argsort(box_group[:, 0, 1])]
        sorted_box_group = np.zeros(box_group.shape)
        lines = []

        # list of indexes
        temp = []
        i = 0

        # check if there is more than one box in the box_group
        if len(box_group) <= 1:
            # since there is only one box in the boxgroup do nothing but copying the box
            return box_group, lines
        while i < len(box_group):
            for j in range(i + 1, len(box_group)):
                if self._is_on_same_line(box_group[i], box_group[j], intersection_thr):
                    if i not in temp:
                        temp.append(i)
                    if j not in temp:
                        temp.append(j)

            # append temp with i if the current box (i) is not on the same line with any other box
            if len(temp) == 0:
                temp.append(i)

            # put boxes on same line into lined_box_group array
            lined_box_group = box_group[np.array(temp)]
            # sort boxes by startX value
            lined_box_group = lined_box_group[np.argsort(lined_box_group[:, 0, 0])]
            lines.append(lined_box_group)

            # skip to the index of the box that is not on the same line
            i = temp[-1] + 1
            # clear list of indexes
            temp = []

        return sorted_box_group, lines

    def sort_bboxes_by_coordinates(self, box_group_my_format: List[BBox], intersection_thr: float = 0.1) -> List:
        box_group = [[[b.x_top_left, b.y_top_left], [b.x_bottom_right, b.y_bottom_right]] for b in box_group_my_format]
        lines = []
        if len(box_group) > 0:
            # sort bounding boxes into lines
            box_group, lines = self._segment_lines(np.array(box_group), intersection_thr)

        res_lines = []
        for line_id, line in enumerate(lines):
            res_lines.append([BBox(box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1]) for box in line])

        return res_lines

    @staticmethod
    def sort_bboxes(img: np.ndarray, bboxes: List[BBox]) -> List[List[BBox]]:
        """
        1) Find separators between textual lines
        2) Move each bbox to the line between separators with the biggest IoU

        :param img: image with all document's words
        :param bboxes: bboxes with text, which were found on the image
        :returns: list of lines of sorted bboxes
        """
        if len(bboxes) == 0:
            return [bboxes]
        x_min, x_max = min([bbox.x_top_left for bbox in bboxes]), max([bbox.x_bottom_right for bbox in bboxes])
        y_min, y_max = min([bbox.y_top_left for bbox in bboxes]), max([bbox.y_bottom_right for bbox in bboxes])
        line_separators = TextDetector._find_lines_separators(img[y_min:y_max, x_min:x_max])
        img_copy = img.copy()
        for line_y in line_separators:
            cv2.line(img_copy, (x_min, line_y), (x_max, line_y), (255, 0, 0), 1)
        cv2.imwrite("image_lines.png", img_copy)
        if len(line_separators) == 0:
            return [bboxes]
        lines_dict = {}
        if line_separators[0] > 0:
            lines_dict[(y_min, y_min + line_separators[0])] = []
        for ind in range(len(line_separators) - 1):
            lines_dict[(y_min + line_separators[ind], y_min + line_separators[ind + 1])] = []
        if y_min + line_separators[-1] < y_max:
            lines_dict[(y_min + line_separators[-1], y_max)] = []

        TextDetector._fill_lines_dict(bboxes, lines_dict)
        lines_list = []
        for key in sorted(lines_dict.keys()):
            value = lines_dict[key]
            if len(value) > 0:
                lines_list.append(sorted(value, key=lambda x: x.x_top_left))
        return lines_list

    def clean_non_text_background(self, image: np.ndarray) -> np.ndarray:
        boxes, confs = self.predict(image)
        return self.delete_background(image, boxes)

    def delete_background(self, image: np.ndarray, boxes: List[BBox]) -> np.ndarray:
        """
        Clean background (clean non-text pixels).
        1) Get mean value of nontext fields on the image
        2) Fill non-text fields by that mean value.

        :param boxes: text bounding boxes
        :return: image with cleaned non-text fields
        """

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image

        mask = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
        for box in boxes:
            mask = cv2.rectangle(mask, (box.x_top_left, box.y_top_left), (box.x_bottom_right, box.y_bottom_right),
                                 color=(0, 0, 0), thickness=-1)

        mask_flatten = mask.flatten()
        grey_flatten = grey.flatten()
        values = [val for ind, val in enumerate(grey_flatten) if mask_flatten[ind] > 0]

        bg_value = np.median(values)

        background = np.full((image.shape[0], image.shape[1]), bg_value, dtype=np.uint8)
        bk = cv2.bitwise_or(background, background, mask=mask)
        fg = cv2.bitwise_or(grey, grey, mask=cv2.bitwise_not(mask))
        image_wo_background = cv2.bitwise_or(bk, fg)

        if self.config.get("debug_mode", False):
            os.makedirs(os.path.join(self.config.get("path_debug"), "text_localization"), exist_ok=True)
            cv2.imwrite(os.path.join(self.config.get("path_debug"), "text_localization", "image_wo_backgound.png"),
                        image_wo_background)

        return image_wo_background

    @staticmethod
    def _fill_lines_dict(bboxes: List[BBox], lines_dict: dict) -> None:
        for bbox in bboxes:
            max_iou, line_key = 0, None
            for y1, y2 in lines_dict.keys():
                min_y1, max_y1 = min(bbox.y_top_left, y1), max(bbox.y_top_left, y1)
                min_y2, max_y2 = min(bbox.y_bottom_right, y2), max(bbox.y_bottom_right, y2)
                iou = (min_y2 - max_y1) / (max_y2 - min_y1)
                if iou > max_iou:
                    max_iou = iou
                    line_key = (y1, y2)
            assert (line_key is not None)
            lines_dict[line_key].append(bbox)

    @staticmethod
    def _find_lines_separators(img: np.ndarray) -> np.ndarray:
        """
        Find y-coordinates for lines which separate document's textual lines
        Example:
            y1 --------------
            First textual line
            y2 --------------
            Second textual line
            ....
        Returns the array of [y1, y2, ...]
        """
        binarized_img = binarize(img)
        projection = binarized_img.sum(axis=1).sum(axis=1)
        projection_smoothed = savgol_filter(projection, 71, 3)
        projection_smoothed = savgol_filter(projection_smoothed, 141, 3)
        peaks, _ = find_peaks(projection_smoothed, height=0)
        return peaks
