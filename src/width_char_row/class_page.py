import os
import numpy as np
import cv2
from typing import List
import pickle
import time

from width_char_row.bbox import BBox
from width_char_row.text_detector import TextDetector
from width_char_row.bold_classifier import PsBoldClassifier, MeanBoldClassifier, TYPE_LINE, TYPE_WORD, TYPE_LINE_WORD
from width_char_row.my_binar import binarize

OFFSET_ROW = 2
BOLD_ROW = 1
REGULAR_ROW = 0

WIDTH = 700
COLOR_BOLD_ROW = (255, 0, 0)
COLOR_OFFSET_ROW = (0, 0, 255)
COLOR_REGULAR_ROW = (0, 255, 0)
TEXT_IMG = 1.2
T_BINARY = 200


class Pages:
    def __init__(self, imgs_path: str):

        files = os.listdir(imgs_path)
        self.pages = []
        self.name_pages = []
        for f in files:
            if f.split(".")[-1] in ["jpg", "png", "jpeg"]:
                pkl_path = os.path.join(imgs_path, f + ".pkl")
                if os.path.isfile(pkl_path):
                    self.name_pages.append(f)
                    path_img = os.path.join(imgs_path, f)
                    self.pages.append(Page(path_img))

    def test_method(self, method, k, print_rez=True, type_stat=TYPE_LINE_WORD, binary_N=5):
        N = len(self.pages)
        count_word = 0
        precision = 0
        recall = 0
        time_work = 0
        cpu_work_time = 0
        for i in range(N):
            count_word_i_page = self.pages[i].count_word()
            count_word += count_word_i_page

            start_time = time.time()
            cpu_start_time = time.process_time()
            self.pages[i].processing_method(method=method, k=k, type_stat=type_stat, binary_N=binary_N)
            style_method = self.pages[i].style
            end_time = time.time()
            cpu_end_time = time.process_time()

            estimation_i_page = self.pages[i].estimation(style_method)
            precision += count_word_i_page*estimation_i_page["precision"]
            recall += count_word_i_page*estimation_i_page["recall"]

            dt = end_time-start_time
            cpu_dt = cpu_end_time-cpu_start_time
            time_work += dt
            cpu_work_time += cpu_dt
            if print_rez:
                print('================================================')
                print(self.name_pages[i])
                print(f"precision:{estimation_i_page['precision']:.4f}")
                print(f"recall{estimation_i_page['recall']:.4f}")
                print(f"Время работы:{dt:.4f} cек (CPU время:{cpu_dt:.4f} сек)")
                print('================================================')
        if print_rez:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(f"precision: {precision/count_word:.4f}")
                print(f"recall: {recall/count_word:.4f}")
                print(f"Время работы:{time_work:.4f} cек (CPU время:{cpu_work_time:.4f} сек)")
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        rez = {
            "precision": precision/count_word,
            "recall": recall/count_word
        }
        return rez


class Page:
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.img = self._read_img(img_path)

        self.lines = []
        self.style = []
        self.coef = []

        self._segmentation_lines()

    def count_word(self):
        rez = 0
        for line in self.lines:
            rez += len(line)
        return rez

    def _read_img(self, name_file: str):
        """
        Открывает изображения cv2 в которых есть кириллические символы
        """
        with open(name_file, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return img

    def _segmentation_lines(self):
        # HOT to use
        '''
        checkpoint_path = "путь до папки с весами для модели детекции"
        image: np.ndarray
        '''

        pkl_path = self.img_path + ".pkl"
        if os.path.exists(pkl_path):
            self.load_segm()
        else:
            checkpoint_path = r"C:\Users\danii\Dropbox\Работа\03.Исходники_программ\веса"
            # 3 - segmentation words into lines

            text_detector = TextDetector(on_gpu=False, checkpoint_path=checkpoint_path, with_vertical_text_detection=False,
                                         config={})
            # 2 - detect text
            boxes, conf = text_detector.predict(self.img)
            lines = text_detector.sort_bboxes_by_coordinates(boxes)
            self.lines = self.union_lines(lines)

    def save_segm(self):
        pkl_path = self.img_path + ".pkl"
        with open(pkl_path, 'wb') as f:
            dict_lines = [[box.to_dict() for box in line] for line in self.lines]
            pickle.dump((dict_lines, self.style), f)

    def load_segm(self):
        pkl_path = self.img_path + ".pkl"
        with open(pkl_path, 'rb') as f:
            (dict_lines, style) = pickle.load(f)
        self.lines = [[BBox.from_dict(box) for box in line] for line in dict_lines]
        self.style = style

    @staticmethod
    def union_lines(lines: List[List[BBox]]) -> List[List[BBox]]:
        filtered_lines = []
        one_id = 0
        while one_id < len(lines) - 1:
            one_line = lines[one_id]
            two_line = lines[one_id + 1]
            merge_line = one_line + two_line
            min_h_one = min([box.y_bottom_right - box.y_top_left for box in one_line])
            min_h_two = min([box.y_bottom_right - box.y_top_left for box in two_line])

            one_bottom = max([box.y_bottom_right for box in one_line])
            two_top = min([box.y_top_left for box in two_line])

            interval_between_lines = two_top - one_bottom
            if interval_between_lines < 0 or (min_h_one > min_h_two and interval_between_lines < min_h_two / 2):
                union_line = sorted(merge_line, key=lambda x: x.x_top_left)
                filtered_lines.append(union_line)
                one_id += 2
            else:
                filtered_lines.append(one_line)
                one_id += 1

        return filtered_lines

    def processing_method(self, k, method, type_stat=TYPE_LINE_WORD):
        if method == "ps":
            ps_classifier = PsBoldClassifier(k, type_stat)
            self.style = ps_classifier.classify(self.img, self.lines)
            self.coef = ps_classifier.get_lines_estimates(self.img, self.lines)
        elif method == "mean":
            ps_classifier = MeanBoldClassifier(k, type_stat)
            self.style = ps_classifier.classify(self.img, self.lines)
            self.coef = ps_classifier.get_lines_estimates(self.img, self.lines)
        else:
            pass

    def imshow(self, binary=False):
        h = self.img.shape[0]
        w = self.img.shape[1]

        if not binary:
            img_cope = self.img.copy()
        else:
            img_cope = binarize(self.img)
        coef = h / w
        exist_style = len(self.style) != 0
        exist_coef = len(self.coef) != 0
        for i in range(len(self.lines)):
            for j in range(len(self.lines[i])):
                border = 1
                word = self.lines[i][j]
                color = (155, 155, 155)
                info_word = ""
                if exist_style:
                    style_word = self.style[i][j]
                    if exist_coef:
                        info_word = self.coef[i][j]

                    if style_word == BOLD_ROW:
                        color = COLOR_BOLD_ROW
                    elif style_word == OFFSET_ROW:
                        color = COLOR_OFFSET_ROW
                    elif style_word == REGULAR_ROW:
                        color = COLOR_REGULAR_ROW
                cv2.rectangle(img_cope, (word.x_top_left, word.y_top_left),
                              (word.x_bottom_right, word.y_bottom_right), color, border)
                cv2.putText(img_cope, f"{info_word:.2f}", (word.x_top_left, word.y_top_left),
                            cv2.FONT_HERSHEY_COMPLEX, TEXT_IMG, color, 1)
        img = cv2.resize(img_cope, (WIDTH, round(coef * WIDTH)))
        cv2.imshow("img", img)
        cv2.waitKey(0)

    def estimation(self, style_method):
        self.load_segm()
        style_true = self.style
        bold_all = 0
        regular_all = 0
        bold_right = 0
        regular_right = 0
        for i in range(len(style_true)):
            for j in range(len(style_true[i])):
                if style_true[i][j] == BOLD_ROW:
                    bold_all += 1
                    if style_method[i][j] == BOLD_ROW:
                        bold_right += 1
                elif style_true[i][j] == REGULAR_ROW:
                    regular_all += 1
                    if style_method[i][j] == REGULAR_ROW:
                        regular_right += 1

        if bold_right+regular_all-regular_right == 0:
            precision = 0
        else:
            precision = bold_right / (bold_right + regular_all - regular_right)
        if bold_all == 0:
            recall = 1
        else:
            recall = min([1, bold_right/bold_all])
        rez = {
            "precision": precision,
            "recall": recall
        }
        return rez



# class WidthCharImage:
#     def __init__(self, img_binary: np.ndarray):
#         # self.img = img
#         self.img_binary = img_binary # self._get_binary_img(T_BINARY)
#         self.h_start, self.h_end, self.h = self._get_h_row()
#
#     def _get_binary_img(self, T_binary: int):
#         binary_img = binarize(self.img)
#         # binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
#         # binary_img[binary_img < T_binary] = 0
#         # binary_img[binary_img >= T_binary] = 1
#         return binary_img
#
#     def _get_h_row(self, permissible_h: int = 5):
#         h = self.img_binary.shape[0]
#         if h < permissible_h:
#             return None
#         mean_ = self.img_binary.mean(1)
#         dmean = abs(mean_[:-1] - mean_[1:])
#
#         max1 = 0
#         max2 = 0
#         argmax1 = 0
#         argmax2 = 0
#         for i in range(len(dmean)):
#             if dmean[i] > max2:
#                 if dmean[i] > max1:
#                     max2 = max1
#                     argmax2 = argmax1
#                     max1 = dmean[i]
#                     argmax1 = i
#                 else:
#                     max2 = dmean[i]
#                     argmax2 = i
#         h_min = min(argmax1, argmax2)
#         h_max = max(argmax1, argmax2)
#         h = h_max - h_min + 1
#
#         return h_min, h_max, h
#
#     def get_h_row(self, binary=True):
#         if binary:
#             return self.img_binary[self.h_start:self.h_end + 1, :]
#         else:
#             return self.img[self.h_start:self.h_end + 1, :, :]
#
#     def get_width_char_row(self, method="mean"):
#         if method == "mean":
#             img_chars = self.get_h_row()
#             x = img_chars.mean(0)
#             x[1:-1] = 1/3*(x[1:-1] + x[:-2] + x[2:])
#             img_chars_without_space = img_chars[:, x < 0.95]
#             return img_chars_without_space.mean()
#
#         elif method == "sq":
#             img_chars = self.get_h_row()
#             rez_sq = self._sq(img_chars, 6)
#             if rez_sq is None:
#                 rez_sq = 0
#             return rez_sq
#
#         elif method == "ps":
#             img_chars = self.get_h_row()
#             x = img_chars.mean(0)
#             img_chars_without_space = img_chars[:, x < 0.95]
#             hw = img_chars_without_space.shape[0]*img_chars_without_space.shape[1]
#             p_img = img_chars[:, :-1] - img_chars[:, 1:]
#             p_img[p_img > 0] = 1
#             p = p_img.sum()
#             s = hw-img_chars_without_space.sum()
#             return p/s
#
#     def _sq(self, img, count_line):
#         delta_h = self.h / (count_line + 1)
#         if img is None:
#             return 0
#         if count_line > self.h:
#             count_line = round(self.h / 3)
#         rez_w = []
#         rez_cord = []
#         for i in range(count_line):
#             width_sq_line_i, cord_sq_line_i = self._sq_one_line(img, round(delta_h*(i+1)))
#             rez_w = rez_w + width_sq_line_i
#             rez_cord = rez_cord + cord_sq_line_i
#
#         while True:
#             if len(rez_w) == 0:
#                 return None
#             index_max = np.argmax(rez_w)
#             r = (np.max(rez_w)-1)/2
#             y0, x0 = rez_cord[index_max]
#             y1 = round(y0-r/1.44)
#             y2 = round(y0+r/1.44)
#             x1 = round(x0-r/1.44)
#             x2 = round(x0+r/1.44)
#             if y2 >= self.h or y1 < 0 or x2 >= img.shape[1] or x1 < 0:
#                 rez_w.pop(index_max)
#                 rez_cord.pop(index_max)
#             elif img[y1, x1] == 1 or img[y1, x2] == 1 or img[y2, x1] == 1 or img[y2, x2] == 1:
#                 rez_w.pop(index_max)
#                 rez_cord.pop(index_max)
#             else:
#                 return (r*2+1)/self.h
#
#     def _sq_one_line(self, img, i_line):
#         x = img[i_line, :]
#         width_sq_line = []
#         cord_sq_line = []
#         is_border = True
#         width_char = 0
#         for i in range(len(x)):
#             if img[i_line, i] == 0:
#                 if is_border:
#                     is_border = False
#                 width_char += 1
#             else:
#                 if not is_border:
#                     is_border = True
#                     center_cord_char = i - round(width_char/2)
#                     width_char = self._one_sq_size(img, i_line, center_cord_char, width_char)
#                     width_sq_line.append(width_char)
#                     cord_sq_line.append([i_line, center_cord_char])
#                     width_char = 0
#
#         return width_sq_line, cord_sq_line
#
#     def _one_sq_size(self, img, i, j, w):
#         k = 0
#         width_char_ = 0
#         while img[i+k, j] != 1:
#             width_char_ += 1
#             k += 1
#             if width_char_ > w:
#                 return w
#             if i+k >= img.shape[0]:
#                 break
#         k = 0
#         width_char_ -= 1
#         while img[i-k, j] != 1:
#             width_char_ += 1
#             k += 1
#             if width_char_ > w:
#                 return w
#             if i-k <= 0:
#                 break
#
#         return width_char_


