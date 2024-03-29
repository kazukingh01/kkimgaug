import cv2
import numpy as np
from typing import List
from kkimgaug.util.functions import fit_resize, check_type_list, convert_1d_array


__all__ = [
    "visualize",
]


COLORS = [
    (255,  0,  0), # Red
    (  0,204,255), # Sky Blue
    (  0,255,  0), # Green
    (255,102,  0), # Orange
    (153, 51,102), # Purple
    (192,192,192), # Gray
    (  0,  0,  0), # Black
]
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(
    img: np.ndarray, bbox: (float, float, float, float), 
    class_name: str, color: (int,int,int)=(0,0,0), thickness: int=2
) -> np.ndarray:
    """
    Visualizes a single bounding box on the image.
    Params::
        img: np.ndarray
        bbox: (float, float, float, float). x_min, y_min, w, h
        class_name: str. class name
        color: RGB color
        thickness: bbox outline thickness.
    """
    img = img.copy()
    font_scale = 0.4
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    img = cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(1.3 * text_height)), color=color, thickness=-1)
    img = cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min + int(1.0 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize_mask(
    img: np.ndarray, mask: np.ndarray, 
    color: (int,int,int)=(0,0,0)
) -> np.ndarray:
    """
    Visualizes a single mask on the image.
    Params::
        img: np.ndarray
        mask: np.ndarray. boolean
        color: color
    """
    img   = img.copy()
    mask  = (mask.copy().astype(np.uint8) > 0)
    imgwk = np.zeros((*mask.shape, 3)).astype(np.uint8)
    for i, _color in enumerate(color):
        imgwk[:, :, i][mask] = _color
    imgwk = imgwk.astype(np.uint8)
    img = cv2.addWeighted(img, 1, imgwk, 0.6, 0)
    return img


def visualize_keypoint(
    img: np.ndarray, keypoint: (float, float),
    class_name: str=None,
    color: (int,int,int)=(0,0,0), diameter=5
):
    """
    Visualizes keypoints on the image.
    Params::
        img: np.ndarray
        keypoint: (x, y)
        class_name: str. 
        color: color
    """
    img = img.copy()
    font_scale = 0.4
    x, y = int(keypoint[0]), int(keypoint[1])
    img = cv2.circle(img, (x, y), diameter, color, -1)
    if class_name:
        ((_, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        img = cv2.putText(
            img,
            text=class_name,
            org=(x, y + int(1.5 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, 
            color=color, 
            lineType=cv2.LINE_AA,
        )
    return img


def visualize(
    img: np.ndarray, 
    bboxes: List[List[float]]=None,
    class_names: List[str]=None,
    class_names_saved: List[str]=None,
    mask: np.ndarray=None,
    keypoints: List[List[float]]=None, 
    class_names_kpt: List[str]=None,
    class_names_kpt_saved: List[str]=None,
    resize: int=None
):
    """
    Visualizes annotations on the image.
    Params::
        img: np.ndarray[uint8]
        bboxes: list of bboxes. [[x_min1, y_min1, w1, h1], [x_min2, y_min2, w2, h2], ..]
        class_names: list of object index. Corresponding to the bboxes. [1, 2, 3, ...]
        class_names_saved: list of class names. ["dog", "dog", "cat", ..]
        mask: np.ndarray[int]. not boolean, interger type.
            ex)
            The value 0 is no mask.
            The value 1 is instance(not class) A (class_names[0])
            The value 2 is instance(not class) B (class_names[1])
            [[0,0,0,0],
             [0,2,1,1],
             [0,2,1,1]]
        keypoints: list of keypoints. [[x1, y1], [x2, y2], ...]
        class_names_kpt: list of indexes of kpt. [0, 2, 7, ..]
        class_names_kpt_saved: list of original definition of kpt. [["eye", "nose", "mouth", ...], ["eye", "nose", "mouth", ...], ...]
    """
    assert isinstance(img, np.ndarray)
    if bboxes is not None and len(bboxes) > 0:
        assert check_type_list(bboxes, [list, tuple], [int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
        if class_names is not None:
            assert check_type_list(class_names, [int, np.int8, np.int16, np.int32, np.int64])
            assert len(bboxes) == len(class_names)
        else:
            class_names = list(range(len(bboxes)))
        if class_names_saved is not None:
            assert check_type_list(class_names_saved, [int, str])
    if mask is not None:
        assert isinstance(mask, np.ndarray)
        check_sum = sum([(mask == x).sum() for x in ([0,] + class_names)])
        # It requires that almost cell (80%) is filled by each label index (including "0" label meaning bachground). Some class is deleted since its size is too small
        assert ((mask.shape[0] * mask.shape[1]) * 0.8) <= check_sum 
        if class_names is None:
            class_names = list(range(len(bboxes)))
    if keypoints is not None and len(keypoints) > 0:
        assert check_type_list(keypoints, [list, tuple], [int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
        for x in keypoints: assert len(x) == 2
        if class_names_kpt is not None:
            assert check_type_list(class_names_kpt, [int, np.int8, np.int16, np.int32, np.int64])
            assert len(class_names_kpt) == len(keypoints)
        else:
            class_names_kpt = list(range(len(keypoints)))
        if class_names_kpt_saved is not None:
            assert check_type_list(class_names_kpt_saved, [list, tuple], [int, str])
            class_names_kpt_saved = convert_1d_array(class_names_kpt_saved)
    img = img.copy()
    dict_color     = {x:COLORS[i%len(COLORS)] for i, x in enumerate(np.unique(class_names_saved    ))} if class_names_saved else None
    dict_kpt_color = {x:COLORS[i%len(COLORS)] for i, x in enumerate(np.unique(class_names_kpt_saved))} if class_names_kpt_saved else None
    if bboxes is not None and len(bboxes) > 0:
        for bbox, i_label in zip(bboxes, class_names):
            class_name = class_names_saved[i_label - 1] if class_names_saved is not None else i_label
            img = visualize_bbox(img, bbox, str(class_name), color=(dict_color[class_name] if dict_color else (255,0,0)))
    if mask is not None:
        for i_label in class_names:
            class_name = class_names_saved[i_label - 1] if class_names_saved is not None else i_label
            img = visualize_mask(img, (mask == i_label), color=(dict_color[class_name] if dict_color else (255,0,0)))
    if keypoints is not None and len(keypoints) > 0:
        for keypoint, i_label in zip(keypoints, class_names_kpt):
            class_name = class_names_kpt_saved[i_label] if class_names_kpt_saved is not None else i_label
            img = visualize_keypoint(img, keypoint, str(class_name), color=(dict_kpt_color[class_name] if dict_kpt_color else (0,255,0)))
    if isinstance(resize, int):
        img = fit_resize(img, "y", resize)
    cv2.imshow(__name__, img)
    cv2.waitKey(0)
