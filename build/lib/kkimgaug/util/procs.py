import copy
import numpy as np
import cv2

from kkimgaug.util.functions import convert_polygon_to_bool, convert_1d_array, convert_same_dimension


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

def mask_from_bool_to_polygon(transformed: dict, label_name: str="mask"):
    """
    Partams::
        transformed[label_name].
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
    """
    if transformed.get(label_name) is not None and len(transformed[label_name]) > 0:
        masks = transformed[label_name].astype(np.uint8)
        list_polygons = []
        for label in np.sort(np.unique(masks)):
            if label == 0: continue
            mask = (masks == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            list_polygons.append([])
            for ndfwk in contours:
                list_polygons[-1].append(convert_1d_array(ndfwk.tolist()))
        transformed[label_name] = list_polygons
    return transformed

def kpt_from_coco_to_xy(transformed: dict, label_name: str="keypoints", label_name_class: str="label_kpt", is_mask_unvis: bool=True):
    """
    Params::
        transformed[label_name]. 
        [
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of class A
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of class B
        ]
    """
    ndf_bool = None
    if transformed.get(label_name) is not None and len(transformed[label_name]) > 0:
        ndf_kpt = np.concatenate(transformed[label_name], axis=0)
        if ndf_kpt.shape[0] > 0:
            transformed[label_name + "_saved"] = copy.deepcopy(transformed[label_name].copy())
            ndf_kpt = ndf_kpt.reshape(-1, 3)
            ndf = None
            if is_mask_unvis:
                ndf_bool = (ndf_kpt[:, -1] > 0) # visual が 1 or 2 を抽出
                ndf = ndf_kpt[ndf_bool][:, :2].copy()
            else:
                ndf = ndf_kpt[:, :2].copy()
            transformed[label_name] = ndf.copy().tolist()
        else:
            transformed[label_name] = []
    if transformed.get(label_name_class) is not None and transformed[label_name_class]:
        ndf = np.concatenate(transformed[label_name_class], axis=0)
        if ndf.shape[0] > 0:
            transformed[label_name_class] = ndf[ndf_bool].copy().tolist() if is_mask_unvis else ndf.copy().tolist()
        else:
            transformed[label_name_class] = []
    if transformed.get(label_name) is not None and len(transformed[label_name]) > 0 and \
       (transformed.get(label_name_class) is None or len(transformed[label_name_class]) == 0):
        # 適当に埋める
        ndf = np.arange(ndf_kpt.shape[0]).astype(int)
        transformed[label_name_class] = ndf[ndf_bool].copy().tolist() if is_mask_unvis else ndf.copy().tolist()
    return transformed

def restore_kpt_coco_format(transformed: dict):
    if transformed.get("keypoints_saved") is not None:
        ndf_kpt       = np.array(convert_1d_array(transformed["keypoints"])).reshape(-1, 2)
        ndf_kpt_saved = np.array(convert_1d_array(transformed["keypoints_saved"])).reshape(-1, 3)
        ndf_kpt       = np.concatenate([ndf_kpt, ndf_kpt_saved[transformed["label_kpt"]][:, -1].reshape(-1, 1)], axis=1)
        list_kpt = convert_1d_array(ndf_kpt.tolist())
        for kpt_label in np.arange(ndf_kpt_saved.shape[0]):
            if not (int(kpt_label) in transformed["label_kpt"]):
                for _ in range(3): list_kpt.insert(kpt_label*3, 0)
        transformed["keypoints"] = convert_same_dimension(list_kpt, transformed["keypoints_saved"])
    return transformed

def get_applied_augmentations(transformed: dict, drow_on_image: bool=False):
    if transformed.get("replay") is not None:
        def work(list_augs: dict):
            list_ret = []
            for dictwk in list_augs:
                if "applied" in dictwk and dictwk["applied"]:
                    if "transforms" in dictwk:
                        list_ret.append(work(dictwk["transforms"]))
                    else:
                        list_ret.append(dictwk["__class_fullname__"])
            return list_ret
        list_applied = work(transformed["replay"]["transforms"])
        list_applied = convert_1d_array(list_applied)
        transformed["__replay"] = list_applied
        if drow_on_image:
            img = transformed["image"]
            x_min, y_min = 0, 0
            for aug in list_applied:
                font_scale = 0.4
                ((text_width, text_height), _) = cv2.getTextSize(aug, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                img = cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(1.3 * text_height)), color=(255,255,255), thickness=-1)
                img = cv2.putText(
                    img,
                    text=aug,
                    org=(x_min, y_min + int(1.0 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, 
                    color=(0,0,0), 
                    lineType=cv2.LINE_AA,
                )
                y_min += int(1.3 * text_height)
            transformed["image"] = img
    return transformed