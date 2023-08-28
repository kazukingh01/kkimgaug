import copy
import numpy as np
import cv2
from PIL import Image
# local packages
from kkimgaug.config.config import LABEL_NAME_IMAGE, LABEL_NAME_BBOX, LABEL_NAME_BBOX_CLASS, LABEL_NAME_MASK, LABEL_NAME_KPT, LABEL_NAME_KPT_CLASS
from kkimgaug.util.functions import convert_polygon_to_bool, convert_1d_array, bbox_from_mask, check_type_list


__all__ = [
    "to_uint8",
    "bgr2rgb",
    "rgb2bgr",
    "pil2nprgb",
    "npbgr2pil",
    "return_image",
    "check_coco_annotations",
    "bbox_label_auto",
    "bbox_compute_from_mask",
    "mask_inside_bbox",
    "mask_from_polygon_to_bool",
    "mask_from_bool_to_polygon",
    "kpt_label_auto",
    "kpt_from_coco_to_xy",
    "kpt_from_xy_to_coco",
    "get_applied_augmentations",
]


def to_uint8(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE):
    """
    Sometimes, output of image has float type. So in case of it, this process run.
    """
    if transformed.get(label_name_image) is not None:
        transformed[label_name_image] = transformed[label_name_image].astype(np.uint8)
    return transformed

def bgr2rgb(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE):
    if transformed.get(label_name_image) is not None:
        transformed[label_name_image] = cv2.cvtColor(transformed[label_name_image], cv2.COLOR_BGR2RGB)
    return transformed

def rgb2bgr(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE):
    if transformed.get(label_name_image) is not None:
        transformed[label_name_image] = cv2.cvtColor(transformed[label_name_image], cv2.COLOR_RGB2BGR)
    return transformed

def pil2nprgb(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE):
    if transformed.get(label_name_image) is not None:
        transformed[label_name_image] = np.array(transformed[label_name_image].convert("RGB"))[:, :, ::-1]
    return transformed

def npbgr2pil(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE):
    if transformed.get(label_name_image) is not None:
        transformed[label_name_image] = Image.fromarray(transformed[label_name_image])
    return transformed

def return_image(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE):
    return transformed[label_name_image]

def check_coco_annotations(
    transformed: dict, label_name_bbox: str=LABEL_NAME_BBOX, label_name_bbox_class: str=LABEL_NAME_BBOX_CLASS, 
    label_name_mask: str=LABEL_NAME_MASK, label_name_kpt: str=LABEL_NAME_KPT, label_name_kpt_class: str=LABEL_NAME_KPT_CLASS
):
    """
    Check annotations of coco format.
    Example of 'transformed'::
        >>> transformed["bboxes"]
        [[50,50,50,50], [200,100,100,100]]
        >>> transformed["label_bbox"]
        ["test1", "test2"]
        >>> transformed["mask"]
        [[[50,50,75,50,75,75,50,75], [75,75,100,75,100,100,75,100]], [[200,100,250,100,250,150,200,150], [250,150,300,150,300,200,250,200]]]
        >>> transformed["keypoints"]
        [[60,60,2.0,80,80,2.0], [225,125,2.0, 275,175,2.0]]  # coco format is x,y,v. "v" means if it is hidden.
        >>> transformed["label_kpt"]
        [["point1", "point2"], ["point1", "point2"]]
    """
    if transformed.get(label_name_bbox) is not None and len(transformed[label_name_bbox]) > 0:
        # check for bbox
        assert check_type_list(transformed[label_name_bbox], [list, tuple], [int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
        for x in transformed[label_name_bbox]: assert len(x) == 4
        if transformed.get(label_name_bbox_class) is not None:
            # check for bbox label
            assert check_type_list(transformed[label_name_bbox_class], [int, str])
            assert len(transformed[label_name_bbox]) == len(transformed[label_name_bbox_class])
        if transformed.get(label_name_mask) is not None:
            """
            check for mask. Mask is required as polygon.
            mask annotation format is following format.
            [
                [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance1
                [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance2
                ...
            ]"""
            assert len(transformed[label_name_mask]) > 0
            assert check_type_list(transformed[label_name_mask], [list, tuple], [list, tuple], [int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
            assert len(transformed[label_name_bbox]) == len(transformed[label_name_mask])
            for x in transformed[label_name_mask]:
                for y in x: assert len(y) > 0
            if len(convert_1d_array(transformed[label_name_mask])) == 0:
                del transformed[label_name_mask] # mask key bust be deleted.
        if transformed.get(label_name_kpt) is not None and len(transformed[label_name_kpt]) > 0:
            """
            check for keypoints. kpt is required coco format.
            [
                [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance1
                [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance2
                ...
            ]
            """
            assert check_type_list(transformed[label_name_kpt], [list, tuple], [int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
            assert len(transformed[label_name_bbox]) == len(transformed[label_name_kpt])
            for x in transformed[label_name_kpt]:
                assert len(x) > 0
                assert len(x) % 3 == 0
                assert len(x) == len(transformed[label_name_kpt][0])
            assert np.all(np.isin(np.concatenate(transformed[label_name_kpt]).reshape(-1, 3)[:, -1], [0, 1, 2])) # Visible parameter is required in [0, 1, 2]
            if len(convert_1d_array(transformed[label_name_kpt])) == 0:
                transformed[label_name_kpt] = []
            if transformed.get(label_name_kpt_class) is not None and len(transformed[label_name_kpt_class]) > 0:
                # check for kpt label name
                assert check_type_list(transformed[label_name_kpt_class], [list, tuple], [int, str])
                assert len(transformed[label_name_kpt]) == len(transformed[label_name_kpt_class])
                for x, y in zip(transformed[label_name_kpt], transformed[label_name_kpt_class]):
                    assert len(x) // 3 == len(y)
    return transformed

def bbox_label_auto(transformed: dict, label_name: str=LABEL_NAME_BBOX, label_name_class: str=LABEL_NAME_BBOX_CLASS):
    """
    Define integer label automaticaly, which makes easier identification.
    Output::
        from:
            'label_bbox': ["dog", "dog", "cat", ...]
        to:
            'label_bbox': [1, 2, 3, ...] # 0 doesn't include here. 0 means background.
    """
    if transformed.get(label_name) is not None:
        assert isinstance(transformed[label_name], list)
        if transformed.get(label_name_class) is not None:
            transformed[f"{label_name_class}_saved"] = copy.deepcopy(transformed[label_name_class])
            transformed[label_name_class] = list(range(1, len(transformed[label_name_class])+1))
        else:
            transformed[label_name_class] = list(range(1, len(transformed[label_name])+1))
            transformed[f"{label_name_class}_saved"] = copy.deepcopy(transformed[label_name_class])
    return transformed

def bbox_compute_from_mask(transformed: dict, label_name_bbox: str=LABEL_NAME_BBOX, label_name_bbox_class: str=LABEL_NAME_BBOX_CLASS, label_name_mask: str=LABEL_NAME_MASK):
    """
    Bbox is computed from mask.
    Assume that label_bbox is defined by "bbox_label_auto" function.
    """
    if transformed.get(label_name_bbox) is not None and transformed.get(label_name_mask) is not None:
        assert isinstance(transformed[label_name_mask], np.ndarray)
        assert transformed[label_name_bbox_class] is not None and check_type_list(transformed[label_name_bbox_class], int)
        check_sum = sum([(transformed[label_name_mask] == x).sum() for x in ([0,] + transformed[label_name_bbox_class])])
        assert (transformed[label_name_mask].shape[0] * transformed[label_name_mask].shape[1]) == check_sum # It requires that all cell is filled by each label index (including "0" label meaning bachground)
        transformed[label_name_bbox] = [] # initialize
        for label in transformed[label_name_bbox_class]:
            x, y, w, h = bbox_from_mask((transformed[label_name_mask] == label), format="coco")
            transformed[label_name_bbox].append([x, y, w, h])
    return transformed

def mask_inside_bbox(transformed: dict, label_name_bbox: str=LABEL_NAME_BBOX, label_name_mask: str=LABEL_NAME_MASK):
    """
    In case unexpected processing ( such as rotation ? ), mask is required inside the bboxes.
    """
    if transformed.get(label_name_mask) is not None and transformed.get(label_name_bbox) is not None:
        assert isinstance(transformed[label_name_mask], np.ndarray)
        ndf = np.zeros_like(transformed[label_name_mask]).astype(bool)
        if len(transformed[label_name_bbox]) > 0:
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
        else:
            # After transformed if the result of bboxes is nothing, mask goes also nothing.
            transformed[label_name_mask][~ndf] = 0
    return transformed

def mask_from_polygon_to_bool(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE, label_name: str=LABEL_NAME_MASK):
    """
    Convering mask image from polygon is required by "Albumentation".
    Params::
        transformed["mask"]
        [
            [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance1
            [[x11, y11, x12, y12, x13, y13, ...] <- seg1 , [x21, y21, ...] <- seg2 , ..], <- instance2
            ...
        ]
    """
    if transformed.get(label_name) is not None:
        assert transformed.get(label_name_image) is not None and isinstance(transformed[label_name_image], np.ndarray)
        assert check_type_list(transformed[label_name], [list, tuple], [list, tuple], [int, float, np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
        assert len(transformed[label_name]) < 256 # This method uses being like image so the number of objects must be less than np.uint8
        masks = transformed[label_name]
        mask  = np.zeros_like(transformed[label_name_image][:, :, 0]).astype(np.uint8)
        for i, __mask in enumerate(masks):
            _mask = convert_polygon_to_bool(
                transformed[label_name_image].shape[0],
                transformed[label_name_image].shape[1],
                __mask
            )
            mask[_mask] = i + 1 # This number describes object index. If you have 2 bboxes ["dog", "cat"], "1" means "dog" and "2" means "cat"
        transformed[label_name] = mask
    return transformed

def mask_from_bool_to_polygon(transformed: dict, label_name: str=LABEL_NAME_MASK, label_name_bbox_class: str=LABEL_NAME_BBOX_CLASS, ignore_n_point: int=6):
    """
    Assuming that "label_name_bbox_class" is defined by "bbox_label_auto" function.
    Partams::
        ignore_n_point:
            If the number of polygons is too small, it can be ignored.
    Input::
        transformed["mask"] is like below. 
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]]) # 0 means background, 1 = "dog1", 2 = "dog2", 3 = "cat1", ...
    """
    if transformed.get(label_name) is not None:
        assert isinstance(transformed[label_name], np.ndarray)
        assert transformed.get(label_name_bbox_class) is not None
        assert check_type_list(transformed[label_name_bbox_class], int) and transformed[label_name_bbox_class] == list(range(1, len(transformed[label_name_bbox_class])+1))
        masks = transformed[label_name].astype(int)
        list_polygons = []
        for label in transformed[label_name_bbox_class]:
            list_polygons.append([])
            mask = (masks == label).astype(np.uint8)
            if mask.sum() > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for ndfwk in contours:
                    listwk = ndfwk.reshape(-1).tolist()
                    if len(listwk) < (2 * ignore_n_point): continue
                    list_polygons[-1].append(listwk)
        transformed[label_name] = list_polygons
    return transformed

def kpt_label_auto(transformed: dict, label_name: str=LABEL_NAME_KPT, label_name_class: str=LABEL_NAME_KPT_CLASS):
    """
    Define integer label automaticaly, which makes easier identification.
    Output::
        from:
            'label_kpt': [["eye", "nose", "mouse", ...], ["eye", "nose", "mouse", ...], ...]
        to:
            'label_kpt': [0, 1, 2, ...]
    """
    if transformed.get(label_name) is not None:
        assert isinstance(transformed[label_name], list)
        for x in transformed[label_name]:
            assert len(x) > 0
            assert len(x) % 3 == 0
            assert len(x) == len(transformed[label_name][0])
        if transformed.get(label_name_class) is not None:
            assert check_type_list(transformed[label_name_class], [list, tuple], [int, str])
            assert len(transformed[label_name]) == len(transformed[label_name_class])
            for x, y in zip(transformed[label_name], transformed[label_name_class]):
                assert len(x) // 3 == len(y)
            transformed[f"{label_name_class}_saved"] = copy.deepcopy(transformed[label_name_class])
            transformed[label_name_class] = np.arange(len(convert_1d_array(transformed[label_name_class])), dtype=int).tolist()
        else:
            transformed[label_name_class] = np.arange(len(convert_1d_array(transformed[label_name]))//3, dtype=int).tolist()
            transformed[f"{label_name_class}_saved"] = [[y for y in range(len(x)//3)] for x in transformed[label_name]]
    return transformed

def kpt_from_coco_to_xy(transformed: dict, label_name: str=LABEL_NAME_KPT, label_name_class: str=LABEL_NAME_KPT_CLASS, is_mask_unvis: bool=True):
    """
    Running "kpt_label_auto" is requied.
    Params::
        is_mask_unvis:
            If the visible value is "1", which means this point is unvisible.
            You want to remove this point or not ?
    Input::
        transformed["keypoints"]. 
        [
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance1
            [x1, y1, v1, x2, y2, v2, ...], <- keypoints of instance2
        ]
    """
    ndf_bool = None
    if transformed.get(label_name) is not None and len(transformed[label_name]) > 0:
        for x in transformed[label_name]:
            assert len(x) > 0
            assert len(x) % 3 == 0
            assert len(x) == len(transformed[label_name][0])
        transformed[f"{label_name}_saved"] = copy.deepcopy(transformed[label_name])
        ndf_kpt  = np.concatenate(transformed[label_name], axis=0).reshape(-1, 3)
        ndf_bool = (ndf_kpt[:, -1] == 2) if is_mask_unvis else np.ones(ndf_kpt.shape[0]).astype(bool)
        ndf_kpt  = ndf_kpt[ndf_bool]
        if transformed.get(label_name_class) is not None and len(transformed[label_name_class]) > 0:
            assert check_type_list(transformed[label_name_class], [int, np.int8, np.int16, np.int32, np.int64])
            transformed[label_name_class] = np.array(transformed[label_name_class])[ndf_bool].tolist()
        transformed[label_name] = ndf_kpt[:, :2].copy().tolist()
    return transformed

def kpt_from_xy_to_coco(transformed: dict, label_name: str=LABEL_NAME_KPT, label_name_class: str=LABEL_NAME_KPT_CLASS):
    if transformed.get(label_name) is not None:
        assert transformed.get(f"{label_name      }_saved") is not None
        assert transformed.get(f"{label_name_class}_saved") is not None
        assert len(transformed[label_name]) == len(transformed[label_name_class])
        list_kpt = []
        ndf_kpt  = np.concatenate(transformed[f"{label_name}_saved"], axis=0).reshape(-1, 3)
        for i, ins in enumerate(transformed[f"{label_name_class}_saved"]):
            list_kpt_ins = []
            for j, _ in enumerate(ins):
                index = (i * len(ins)) + j
                if index in transformed[label_name_class]:
                    index_tmp     = np.where(np.array(transformed[label_name_class]) == index)[0][0]
                    list_kpt_ins += (list(transformed[label_name][index_tmp]) + [ndf_kpt[index, -1], ])
                else:
                    list_kpt_ins += [0, 0, 0]
            list_kpt.append([float(x) for x in list_kpt_ins])
        transformed[label_name      ] = list_kpt
        transformed[label_name_class] = transformed[f"{label_name_class}_saved"]
    return transformed

def get_applied_augmentations(transformed: dict, label_name_image: str=LABEL_NAME_IMAGE, draw_on_image: bool=False):
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
        if draw_on_image:
            img = transformed[label_name_image]
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
            transformed[label_name_image] = img
    return transformed