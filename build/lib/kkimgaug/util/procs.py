import numpy as np
import cv2

from kkimgaug.util.functions import convert_polygon_to_bool


def to_uint8(transformed: dict):
    """
    詳細は分からないが、imageがfloatで出力されるAugmentationがある
    """
    if transformed.get("image") is not None:
        transformed["image"] = transformed["image"].astype(np.uint8)
    return transformed

def bgr2rgb(transformed: dict):
    if transformed.get("image") is not None:
        transformed["image"] = cv2.cvtColor(transformed["image"], cv2.COLOR_BGR2RGB)
    return transformed

def rgb2bgr(transformed: dict):
    if transformed.get("image") is not None:
        transformed["image"] = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
    return transformed

def mask_from_polygon_to_bool(transformed: dict, label_name: str="mask"):
    """
    Params::
        transformed[label_name]. 
        [
            [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- class A
            [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- class B
        ]
    """
    if transformed.get(label_name) is not None and transformed[label_name]:
        masks = transformed[label_name]
        mask  = None
        for i, __mask in enumerate(masks):
            _mask = convert_polygon_to_bool(
                transformed["image"].shape[0],
                transformed["image"].shape[1],
                __mask
            )
            if mask is None:
                mask = _mask.copy().astype(np.uint8)
            else:
                mask[_mask] = i + 1
        transformed[label_name] = mask
    return transformed

def kpt_from_coco_to_xy(transformed: dict, label_name: str="keypoints", label_name_class: str="label_kpt"):
    """
    Params::
        transformed[label_name]. 
        [
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of class A
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of class B
        ]
    """
    ndf_bool = None
    if transformed.get(label_name) is not None and transformed[label_name]:
        ndf = np.concatenate(transformed[label_name], axis=0)
        if ndf.shape[0] > 0:
            ndf = ndf.reshape(-1, 3)
            ndf_bool = (ndf[::, -1] > 0) # visual が 1 or 2 を抽出
            ndf = ndf[ndf_bool][:, :2]
            transformed[label_name] = ndf.copy().tolist()
        else:
            transformed[label_name] = []
    if transformed.get(label_name_class) is not None and transformed[label_name_class] and ndf_bool is not None:
        ndf = np.concatenate(transformed[label_name_class], axis=0)
        if ndf.shape[0] > 0:
            transformed[label_name_class] = ndf[ndf_bool].copy().tolist()
        else:
            transformed[label_name_class] = []
    return transformed