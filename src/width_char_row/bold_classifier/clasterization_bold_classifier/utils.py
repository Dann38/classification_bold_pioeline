import numpy as np

PERMISSIBLE_H_BBOX = 5


def get_rid_spaces(image: np.ndarray) -> np.ndarray:
    x = image.mean(0)
    return image[:, x < 0.95]


def base_line_image(image: np.ndarray) -> np.ndarray:
    h = image.shape[0]
    if h < PERMISSIBLE_H_BBOX:
        return image
    mean_ = image.mean(1)
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

    return image[h_min:h_max + 1, :]

