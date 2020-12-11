import os, sys, cv2, copy
import numpy as np
from typing import List
import more_itertools as itr

def get_args() -> dict:
    dict_ret = {}
    args = sys.argv
    dict_ret["__fname"] = args[0]
    for i, x in enumerate(args):
        if   x[:4] == "----":
            # この引数の後にはLISTで格納する
            dict_ret[x[4:]] = []
            for _x in args[i+1:]:
                if _x[:2] != "--": dict_ret[x[4:]].append(_x)
                else: break
        elif x[:3] == "---":
            dict_ret[x[3:]] = True
        elif x[:2] == "--":
            dict_ret[x[2:]] = args[i+1]
    return dict_ret

def correct_dirpath(dirpath: str) -> str:
    if os.name == "nt":
        return dirpath if dirpath[-1] == "\\" else (dirpath + "\\")
    else:
        return dirpath if dirpath[-1] == "/" else (dirpath + "/")

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