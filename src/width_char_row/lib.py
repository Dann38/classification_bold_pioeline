import cv2
import numpy

def read_img(name_file):
    """
    Открывает изображения cv2 в которых есть кириллические символы
    """
    with open(name_file, "rb") as f:
        chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
    return img
