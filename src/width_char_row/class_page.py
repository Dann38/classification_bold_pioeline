import os
import numpy as np
import cv2
import pytesseract
from width_char_row.binarization import binarize
OFFSET_ROW = 2
BOLD_ROW = 1
REGULAR_ROW = 0

WIDTH = 700
COLOR_BOLD_ROW = (255, 0, 0)
COLOR_OFFSET_ROW = (0, 0, 255)
COLOR_REGULAR_ROW = (0, 255, 0)
TEXT_IMG = 2

class Pages:
    def __init__(self, imgs_path: str):

        files = os.listdir(imgs_path)
        self.pages = []
        self.name_pages = []
        for f in files:
            npy_path = os.path.join(imgs_path, f + ".npy")
            if os.path.isfile(npy_path):
                self.name_pages.append(f)
                path_img = os.path.join(imgs_path, f)
                self.pages.append(Page(path_img))

    def test_method(self, method, k, print_rez=True):
        N = len(self.pages)
        count_row = 0
        precision = 0
        recall = 0
        for i in range(N):
            count_row_i_page = self.pages[i].get_count_row()
            count_row += count_row_i_page
            style_i_page = self.pages[i].get_type_rows(method=method, k=k)
            estimation_i_page = self.pages[i].estimation(style_i_page)
            precision += count_row_i_page*estimation_i_page["precision"]
            recall += count_row_i_page*estimation_i_page["recall"]
            if print_rez:
                print('================================================')
                print(self.name_pages[i])
                print(estimation_i_page)
                print('================================================')
        if print_rez:
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print("precision:", precision/count_row)
                print("recall", recall/count_row)
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        rez = {
            "precision": precision/count_row,
            "recall": recall/count_row
        }
        return rez


class Page:
    def __init__(self, img_path: str, T_binary: int = 125):
        npy_path = img_path + ".npy"
        self.img = self._read_img(img_path)
        # self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gray_img = self._get_binary_img(T_binary)
        self.exist_markup = True

        if os.path.isfile(npy_path):
            with open(npy_path, 'rb') as f:
                self.box = np.load(f).tolist()
        else:
            self.exist_markup = False

            box = pytesseract.image_to_data(self.gray_img, output_type=pytesseract.Output.DICT)
            n_boxes = len(box["text"])
            for i in range(n_boxes):
                x0, y0, h0, w0 = int(box['left'][i]), int(box['top'][i]), int(box['height'][i]), int(box['width'][i])
                if (box["level"][i] == 4) and (h0 > 5) and (w0 > 5):
                    box.append([REGULAR_ROW, x0, y0, h0, w0])
            self.box = np.array(box)

    def get_count_row(self):
        return len(self.box)

    def _get_binary_img(self, T_binary: int):
        binary_img = binarize(self.img)
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        binary_img[binary_img < T_binary] = 0
        binary_img[binary_img >= T_binary] = 1
        return binary_img

    def _read_img(self, name_file: str):
        """
        Открывает изображения cv2 в которых есть кириллические символы
        """
        with open(name_file, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return img

    def get_rows(self):
        img_rows = self.get_images_row(img_type="brg")
        gray_img_rows = self.get_images_row(img_type="binary")
        rows = []
        for i in range(len(img_rows)):
            rows.append(Row(img_rows[i], gray_img_rows[i]))
        return rows

    def get_images_row(self, img_type="binary"):
        images_row = []
        if img_type == "brg":
            img_ = self.img
        elif img_type == "binary":
            img_ = self.gray_img

        for j in range(len(self.box)):
            f, x1, y1, h1, w1 = self.box[j]
            row = img_[y1:y1 + h1, x1:x1 + w1]
            images_row.append(row)
        return images_row

    def get_width_rows(self, method="mean"):
        rows = self.get_rows()
        coef_rows = []
        for i in range(len(rows)):
            coef_rows.append(rows[i].get_width_char_row(method=method))
        return coef_rows

    def get_type_rows(self, k, method="mean"):
        coef = self.get_width_rows(method=method)
        rez = np.array(coef)
        rez[rez < k] = 0
        rez[rez >= k] = 1
        return 1-rez

    def estimation(self, style):
        x0 = np.array(self.box)[:, 0]
        x1 = style + x0
        bold_all = len(x0[x0 == 1])
        regular_all = len(x0[x0 == 0])
        bold_right = len(x1[x1 == 2])
        regular_right = len(x1[x1 == 0])
        if bold_right+regular_all-regular_right == 0:
            precision = 0
        else:
            precision = bold_right / (bold_right + regular_all - regular_right)
        if bold_all == 0:
            recall = 1
        else:
            recall = min(1, bold_right/bold_all)
        rez ={
            "precision": precision,
            "recall": recall
        }
        return rez

    def imshow(self, style, info_rows=None):
        h = self.img.shape[0]
        w = self.img.shape[1]
        img_cope = self.img.copy()
        coef = h / w

        for i in range(len(self.box)):
            color = (155, 155, 155)
            border = 1
            font, x0, y0, h0, w0 = self.box[i]
            if style[i] == BOLD_ROW:
                color = COLOR_BOLD_ROW
            elif style[i] == OFFSET_ROW:
                color = COLOR_OFFSET_ROW
            elif style[i] == REGULAR_ROW:
                color = COLOR_REGULAR_ROW
            cv2.rectangle(img_cope, (x0, y0), (x0 + w0, y0 + h0), color, border)
            if info_rows is not None:
                cv2.putText(img_cope, f"{info_rows[i]:.2f}", (x0, y0), cv2.FONT_HERSHEY_COMPLEX, TEXT_IMG, color, 1)
        img = cv2.resize(img_cope, (WIDTH, round(coef * WIDTH)))
        cv2.imshow("img", img)
        cv2.waitKey(0)


class Row:
    def __init__(self, img, img_binary):
        self.img = img
        self.img_binary = img_binary
        self.h_start, self.h_end, self.h = self._get_h_row()

    def _get_h_row(self, permissible_h: int = 5):
        h = self.img_binary.shape[0]
        if h < permissible_h:
            return None
        mean_ = self.img_binary.mean(1)
        dmean = abs(mean_[:-1] - mean_[1:])

        max1 = 0
        max2 = 0
        argmax1 = 0
        argmax2 = 0
        for i in range(len(dmean)):
            if dmean[i] > max2:
                if dmean[i] > max1:
                    max2 = max1
                    argmax2 = argmax1
                    max1 = dmean[i]
                    argmax1 = i
                else:
                    max2 = dmean[i]
                    argmax2 = i
        h_min = min(argmax1, argmax2)
        h_max = max(argmax1, argmax2)
        h = h_max - h_min + 1

        return h_min, h_max, h

    def get_h_row(self, binary=True):
        if binary:
            return self.img_binary[self.h_start:self.h_end + 1, :]
        else:
            return self.img[self.h_start:self.h_end + 1, :, :]

    def get_width_char_row(self, method="mean"):
        if method == "mean":
            img_chars = self.get_h_row()
            x = img_chars.mean(0)
            x[1:-1] = 1/3*(x[1:-1] + x[:-2] + x[2:])
            img_chars_without_space = img_chars[:, x < 0.95]
            return img_chars_without_space.mean()

        elif method == "sq":
            img_chars = self.get_h_row()
            return self._sq(img_chars, 6)

        elif method == "ps":
            img_chars = self.get_h_row()
            x = img_chars.mean(0)
            img_chars_without_space = img_chars[:, x < 0.95]
            hw = img_chars_without_space.shape[0]*img_chars_without_space.shape[1]
            p_img = img_chars[:, :-1] - img_chars[:, 1:]
            p_img[p_img > 0] = 1
            p = p_img.sum()
            s = hw-img_chars_without_space.sum()
            return p/s

    def _sq(self, img, count_line):
        delta_h = self.h / (count_line + 1)
        if img is None:
            return 0
        if count_line > self.h:
            count_line = round(self.h / 3)
        rez_w = []
        rez_cord = []
        for i in range(count_line):
            width_sq_line_i, cord_sq_line_i = self._sq_one_line(img, round(delta_h*(i+1)))
            rez_w = rez_w + width_sq_line_i
            rez_cord = rez_cord + cord_sq_line_i

        while True:
            if len(rez_w) == 0:
                return None
            index_max = np.argmax(rez_w)
            r = (np.max(rez_w)-1)/2
            y0, x0 = rez_cord[index_max]
            y1 = round(y0-r/1.44)
            y2 = round(y0+r/1.44)
            x1 = round(x0-r/1.44)
            x2 = round(x0+r/1.44)
            if y2 > self.h or y1 < 0 or x2 > img.shape[1] or x1 < 0:
                rez_w.pop(index_max)
                rez_cord.pop(index_max)
            elif img[y1, x1] == 1 or img[y1, x2] == 1 or img[y2, x1] == 1 or img[y2, x2] == 1:
                rez_w.pop(index_max)
                rez_cord.pop(index_max)
            else:
                return (r*2+1)/self.h

    def _sq_one_line(self, img, i_line):
        x = img[i_line, :]
        width_sq_line = []
        cord_sq_line = []
        is_border = True
        width_char = 0
        for i in range(len(x)):
            if img[i_line, i] == 0:
                if is_border:
                    is_border = False
                width_char += 1
            else:
                if not is_border:
                    is_border = True
                    center_cord_char = i - round(width_char/2)
                    width_char = self._one_sq_size(img, i_line, center_cord_char, width_char)
                    width_sq_line.append(width_char)
                    cord_sq_line.append([i_line, center_cord_char])
                    width_char = 0

        return width_sq_line, cord_sq_line

    def _one_sq_size(self, img, i, j, w):
        k = 0
        width_char_ = 0
        while img[i+k, j] != 1:
            width_char_ += 1
            k += 1
            if width_char_ > w:
                return w
            if i+k >= img.shape[0]:
                break
        k = 0
        width_char_ -= 1
        while img[i-k, j] != 1:
            width_char_ += 1
            k += 1
            if width_char_ > w:
                return w
            if i-k <= 0:
                break

        return width_char_