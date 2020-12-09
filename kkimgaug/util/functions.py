import os, cv2
import numpy as np
from typing import List

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
