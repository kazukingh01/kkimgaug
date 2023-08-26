import os, cv2, copy, glob, re
import numpy as np
from typing import List, Union
import more_itertools as itr


__all__ = [
    "correct_dirpath",
    "get_file_list",
    "convert_polygon_to_bool",
    "convert_1d_array",
    "convert_same_dimension",
    "bbox_from_mask",
    "fit_resize",
    "check_type",
    "check_type_list",
]


def correct_dirpath(dirpath: str) -> str:
    if os.name == "nt":
        return dirpath if dirpath[-1] == "\\" else (dirpath + "\\")
    else:
        return dirpath if dirpath[-1] == "/" else (dirpath + "/")

def get_file_list(dirpath: str, regex_list: List[str] = []) -> List[str]:
    dirpath = correct_dirpath(dirpath)
    file_list_org = glob.glob(dirpath + "**", recursive=True)
    file_list     = []
    for regstr in regex_list:
        file_list += list(filter(lambda x: len(re.findall(regstr, x)) > 0, file_list_org))
    return file_list if len(regex_list) > 0 else file_list_org

def convert_polygon_to_bool(img_height: int, img_width: int, segmentations: List[List[float]], outline_only: bool=False) -> np.ndarray:
    """
    polygon 形式 [x1, y1, x2, y2, ..] で囲まれた segmentation の area を boolean に変更する関数
    Params::
        img_height: int, 
        img_width:  int, 
        segmentations: [[x11,y11,x12,y12,...], [x21,y21,x22,y22,...], ...]. ※[seg1, seg2, ...]となっている
    """
    img = np.zeros((int(img_height), int(img_width), 3)).astype(np.uint8)
    img_add = img.copy()
    for seg in segmentations:
        # segmentation を 線を繋いで描く
        ndf = cv2.polylines(img.copy(), [np.array(seg).reshape(-1,1,2).astype(np.int32)], True, (255,255,255))
        if outline_only == False:
            # 線を描いた後、一番外の輪郭を取得する
            contours, _ = cv2.findContours(ndf[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 一番外の輪郭内部を埋める
            ndf = cv2.drawContours(ndf, contours, -1, (255,255,255), -1)
        img_add += ndf
    # boolrean に変更する
    img_add = (img_add[:, :, 0] > 0).astype(bool)
    return img_add

def convert_1d_array(arrays: List[object]):
    """
    Usage::
        >>> convert_1d_array([1,2,3, [[1,1,23],2,3]])
        [1, 2, 3, 1, 1, 23, 2, 3]
    """
    arrays = copy.deepcopy(arrays)
    for i, x in enumerate(arrays):
        if not (isinstance(x, list) or isinstance(x, tuple)):
            arrays[i] = [x]
    arrays = list(itr.flatten(arrays))
    i = 0
    if len(arrays) > 0:
        while(1):
            if isinstance(arrays[i], list) or isinstance(arrays[i], tuple):
                arrays = convert_1d_array(arrays)
                i = 0
            else:
                i += 1
            if len(arrays) == i:
                break
    return arrays

def convert_same_dimension(arrays: List[object], array_like: List[object]):
    """
    Usage::
        >>> convert_same_dimension([0,1,2,3,4,5,6,7,8,9], [1, [1, 2], [1, 2, [1, 2, [0, 0], 3]]])
        [0, [1, 2], [3, 4, [5, 6, [7, 8], 9]]]
    """
    arrays = convert_1d_array(arrays)
    if len(arrays) != len(convert_1d_array(array_like)):
        raise Exception(f'Number of items inside lists is different.')
    def work(_list1, _list2) -> List[object]:
        list_ret = []
        for i, x in enumerate(_list2):
            if isinstance(x, list) or isinstance(x, tuple):
                list_ret.append(
                    work(
                        _list1[
                            len(convert_1d_array(list_ret)):
                            len(convert_1d_array(list_ret))+len(convert_1d_array(x))
                        ], 
                        x
                    )
                )
            else:
                list_ret.append(_list1[len(convert_1d_array(list_ret))])
        return list_ret
    return work(arrays, array_like)

def bbox_from_mask(mask: np.ndarray, format: str="coco") -> (float, float, float, float):
    """
    params::
        mask: np.ndarray[bool]
        format:
            coco: x, y, w, h
            xy: x_min, y_min, x_max, y_max
    """
    y_min = np.where(mask)[0].min()
    y_max = np.where(mask)[0].max()
    x_min = np.where(mask)[1].min()
    x_max = np.where(mask)[1].max()
    if format == "coco":
        return x_min, y_min, x_max - x_min, y_max - y_min
    if format == "xy":
        return x_min, y_min, x_max, y_max

def fit_resize(img: np.ndarray, dim: str, scale: Union[int, float]):
    """
    Params::
        img: image
        dim: x or y
        scale: width or height
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(dim, str) and dim in ["x","y"]
    assert isinstance(scale, int) and scale > 10
    height,       width, _    = img.shape
    height_after, width_after = None, None
    if   dim == "x":
        width_after  = int(scale)
        height_after = int(height * (scale / width))
    elif dim == "y":
        height_after = int(scale)
        width_after  = int(width * (scale / height))
    img = cv2.resize(img , (width_after, height_after)) # w, h
    return img

def check_type(instance: object, _type: Union[object, List[object]]):
    _type = [_type] if not (isinstance(_type, list) or isinstance(_type, tuple)) else _type
    is_check = [isinstance(instance, __type) for __type in _type]
    if sum(is_check) > 0:
        return True
    else:
        return False

def check_type_list(instances: List[object], _type: Union[object, List[object]], *args: Union[object, List[object]]):
    """
    Usage::
        >>> check_type_list([1,2,3,4], int)
        True
        >>> check_type_list([1,2,3,[4,5]], int, int)
        True
        >>> check_type_list([1,2,3,[4,5,6.0]], int, int)
        False
        >>> check_type_list([1,2,3,[4,5,6.0]], int, [int,float])
        True
    """
    if isinstance(instances, list) or isinstance(instances, tuple):
        for instance in instances:
            if len(args) > 0 and isinstance(instance, list):
                is_check = check_type_list(instance, *args)
            else:
                is_check = check_type(instance, _type)
            if is_check == False: return False
        return True
    else:
        return check_type(instances, _type)
