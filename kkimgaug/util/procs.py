import copy
import numpy as np
import cv2

from kkimgaug.util.functions import convert_polygon_to_bool, convert_1d_array, convert_same_dimension, bbox_from_mask


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

def check_coco_annotations(transformed: dict, label_name_bbox: str="bboxes", label_name_mask: str="mask", label_name_kpt: str="keypoints"):
    """
    Check annotations of coco format.
    """
    if transformed.get(label_name_bbox) is not None and len(transformed[label_name_bbox]) > 0:
        if transformed.get(label_name_mask) is not None and len(transformed[label_name_mask]) > 0:
            if len(transformed[label_name_bbox]) != len(transformed[label_name_mask]):
                raise Exception(
                    'bbox and mask annotation length is different. ' + \
                    f'bbox: {len(transformed[label_name_bbox])}, mask: {len(transformed[label_name_mask])}'
                )
            for polygon in transformed[label_name_mask]:
                if not (isinstance(polygon, list) or isinstance(polygon, tuple)):
                    raise Exception("""mask annotation format is following format.
[
    [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance1
    [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance2
    ...
]"""                )
                if len(polygon) == 0:
                    raise Exception(f"polygon length is zero. {transformed[label_name_mask]}")
        if transformed.get(label_name_kpt) is not None and len(transformed[label_name_kpt]) > 0:
            if len(transformed[label_name_bbox]) != len(transformed[label_name_kpt]):
                raise Exception(
                    'bbox and keypoints annotation length is different. ' + \
                    f'bbox: {len(transformed[label_name_bbox])}, mask: {len(transformed[label_name_kpt])}'
                )
            base_length = None
            for keypoints in transformed[label_name_mask]:
                if not (isinstance(keypoints, list) or isinstance(keypoints, tuple)):
                    raise Exception("""keypoints annotation format is following format.
[
    [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance1
    [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance2
    ...
]"""                )
                if base_length is None: base_length = len(keypoints)
                if len(keypoints) != base_length:
                    raise Exception(f"keypoints length is different. {transformed[label_name_mask]}")
    return transformed

def bbox_label_auto(transformed: dict, label_name: str="bboxes", label_name_class: str="label_bbox"):
    """
    Define label auto.
    transforme{
        'bboxes': [(x1, y1, w1, h1), (x2, y2, w2, h2), ...]
    }
    """
    if transformed.get(label_name) is not None and len(transformed[label_name]) > 0 and \
       (transformed.get(label_name_class) is None or len(transformed[label_name_class]) == 0):
        ndf = np.arange(len(transformed[label_name])).astype(int)
        transformed[label_name_class] = ndf.copy().tolist()
    return transformed

def bbox_compute_from_mask(transformed: dict, label_name_bbox: str="bboxes", label_name_bbox_class: str="label_bbox", label_name_mask: str="mask"):
    """
    Bbox is computed from mask.
    Assume that label_bbox is defined by "bbox_label_auto" function.
    """
    if transformed.get(label_name_mask) is not None and isinstance(transformed[label_name_mask], np.ndarray) and \
       transformed.get(label_name_bbox_class) is not None and len(transformed[label_name_bbox_class]) > 0:
        transformed[label_name_bbox] = []
        for label in transformed[label_name_bbox_class]:
            x, y, w, h = bbox_from_mask((transformed[label_name_mask] == (label + 1)), format="coco")
            transformed[label_name_bbox].append([x, y, w, h])
    return transformed

def mask_inside_bbox(transformed: dict, label_name_bbox: str="bboxes", label_name_mask: str="mask"):
    """
    albumentations では 20度とかの rotation 
    """
    if transformed.get(label_name_mask) is not None and isinstance(transformed[label_name_mask], np.ndarray) and \
       transformed.get(label_name_bbox) is not None and len(transformed[label_name_bbox]) > 0:
        ndf = np.zeros_like(transformed[label_name_mask]).astype(bool)
        for x, y, w, h in transformed[label_name_bbox]:
            polygon = np.array([
                x,y,
                x+w,y,
                x+w,y+h,
                x,y+h
            ]).reshape(-1, 2).astype(int)
            polygon[:, 0][polygon[:, 0] <= 0] = 0
            polygon[:, 1][polygon[:, 1] <= 0] = 0
            polygon[:, 0][polygon[:, 0] >= transformed[label_name_mask].shape[1]] = transformed[label_name_mask].shape[1] - 1 
            polygon[:, 1][polygon[:, 1] >= transformed[label_name_mask].shape[0]] = transformed[label_name_mask].shape[0] - 1
            ndfwk = convert_polygon_to_bool(
                transformed[label_name_mask].shape[0], 
                transformed[label_name_mask].shape[1],
                [polygon.reshape(-1).tolist()]
            )
            ndf = (ndf | ndfwk)
        transformed[label_name_mask][~ndf] = 0
    return transformed

def mask_from_polygon_to_bool(transformed: dict, label_name: str="mask"):
    """
    Params::
        transformed[label_name]. 
        [
            [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance1
            [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance2
            ...
        ]
    """
    if transformed.get(label_name) is not None and len(transformed[label_name]) > 0:
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

def mask_from_bool_to_polygon(transformed: dict, label_name: str="mask", label_name_bbox_class: str="label_bbox", ignore_n_point: int=6):
    """
    Assume that label_bbox is defined by "bbox_label_auto" function.
    Partams::
        transformed[label_name].
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
    """
    if transformed.get(label_name) is not None and len(transformed[label_name]) > 0 and \
       transformed.get(label_name_bbox_class) is not None and len(transformed[label_name_bbox_class]) > 0:
        masks = transformed[label_name].astype(int)
        list_polygons = []
        for label in transformed[label_name_bbox_class]:
            list_polygons.append([])
            mask = (masks == (label + 1)).astype(np.uint8)
            if mask.sum() > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for ndfwk in contours:
                    listwk = ndfwk.reshape(-1).tolist()
                    if len(listwk) < 2 * ignore_n_point: continue
                    list_polygons[-1].append(listwk)
        transformed[label_name] = list_polygons
    return transformed

def kpt_from_coco_to_xy(transformed: dict, label_name: str="keypoints", label_name_class: str="label_kpt", is_mask_unvis: bool=True):
    """
    Params::
        transformed[label_name]. 
        [
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance1
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance2
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
        # label が定義されていない場合は適当に埋める
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