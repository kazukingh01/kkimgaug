import json
import numpy as np
import cv2
from typing import Union, List
from functools import partial

# locla package
from kkimgaug.lib import BaseCompose
from kkimgaug.util.visualize import visualize
from kkimgaug.util.procs import bgr2rgb, rgb2bgr, mask_from_polygon_to_bool, kpt_from_coco_to_xy, to_uint8, get_applied_augmentations, \
    check_coco_annotations, bbox_label_auto, mask_inside_bbox, bbox_compute_from_mask
from kkimgaug.util.functions import correct_dirpath, convert_1d_array


__all__ = [
    "CocoVisualizer"
]


class CocoItem:
    def __init__(self, coco: dict):
        self.coco = coco
        self.ndf_annotations = np.array(self.coco["annotations"])
        self.dict_category   = {dictwk["id"]:dictwk for dictwk in self.coco["categories"]}
        self.dict_id_fname = {}
        for dictwk in self.coco["images"]:
            self.dict_id_fname[dictwk["file_name"]] = dictwk["id"]
        self.dict_id_fpath = {}
        for dictwk in self.coco["images"]:
            self.dict_id_fpath[dictwk["coco_url"]] = dictwk["id"]
        self.dict_fname_fpath = {}
        for dictwk in self.coco["images"]:
            self.dict_id_fpath[dictwk["file_name"]] = dictwk["coco_url"]
        self.dict_fname_id = {y:x for x, y in self.dict_id_fname.items()}
        self.dict_fpath_id = {y:x for x, y in self.dict_id_fpath.items()}
        self.list_image_id = []
        for dictwk in self.coco["annotations"]:
            self.list_image_id.append(dictwk["image_id"])
        self.list_image_id = np.array(self.list_image_id)

    def __getitem__(self, item: Union[int, str]) -> List[dict]:
        """
        Params::
            item:
                int or str.
                image id or image name
                return list of annotations
        """
        index = item
        if isinstance(item, str):
            index = self.dict_id_fname[item]
        list_annotations = self.ndf_annotations[self.list_image_id == index].tolist()
        list_categories  = [self.dict_category[dictwk["category_id"]] for dictwk in list_annotations]
        return list_annotations, list_categories
    
    def get_fname_from_id(self, id: int):
        return self.dict_fname_id[id]
    
    def get_fpath_from_id(self, id: int):
        return self.dict_fpath_id[id]

    def get_fpath_from_fname(self, fname: str):
        return self.dict_fname_fpath[fname]


class CocoVisualizer:
    def __init__(
        self,
        config: Union[str, dict],
        coco_json: Union[str, dict],
        image_dir: str=None,
        draw_on_image: bool=True
    ):
        """
        Augmentation Sample Visualize for rcoco format 
        Params::
            config: augmentation config. file path or dictionary
            coco_json: coco format json file path or coco format dictionary
            image_dir:
                images directory path. if None, use coco's "coco_url".
                If not None, image path is "image_dir" + coco's "file_name".
        """
        self.composer = BaseCompose(
            config=config,
            preproc=[
                bgr2rgb, 
                check_coco_annotations,
                bbox_label_auto,
                mask_from_polygon_to_bool,
                kpt_from_coco_to_xy
            ],
            aftproc=[
                rgb2bgr,
                mask_inside_bbox,
                bbox_compute_from_mask,
                partial(get_applied_augmentations, draw_on_image=draw_on_image),
                to_uint8,
            ]
        )
        self.coco = json.load(open(coco_json)) if isinstance(coco_json, str) else coco_json
        self.coco = CocoItem(self.coco)
        self.image_dir = correct_dirpath(image_dir) if image_dir else None
    
    def get_image(self, item: Union[int, str]):
        """
        Params::
            item:
                int or str. image id or image name
        """
        file_path = None
        if   isinstance(item, str) and self.image_dir:
            file_path = self.image_dir + item
        elif isinstance(item, str) and not self.image_dir:
            file_path = self.coco.get_fpath_from_fname(item)
        elif isinstance(item, int) and self.image_dir:
            file_path = self.image_dir + self.coco.get_fname_from_id(item)
        elif isinstance(item, int) and not self.image_dir:
            file_path = self.coco.get_fpath_from_id(item)
        img = cv2.imread(file_path)
        return img
    
    def _show(self, transformed: dict, is_bbox: bool=True, is_mask: bool=True, is_kpt: bool=True):
        visualize(
            transformed["image"],
            bboxes=transformed["bboxes"] if is_bbox and transformed.get("bboxes") is not None else None,
            class_names=transformed["label_bbox"] if is_bbox and transformed.get("label_bbox") is not None else None,
            mask=transformed["mask"] if is_mask and transformed.get("mask") is not None else None,
            keypoints=transformed["keypoints"] if is_kpt and transformed.get("keypoints") is not None else None,
            class_names_kpt=transformed["label_kpt"] if is_kpt and transformed.get("label_kpt") is not None else None
        )
    
    def show(self, item: Union[int, str], is_bbox: bool=True, is_mask: bool=True, is_kpt: bool=True):
        """
        Params::
            item:
                int or str. image id or image name
        """
        img = self.get_image(item)
        list_anns, list_cat = self.coco[item]
        ndf_label_bbox = np.array(convert_1d_array([x["name"] for x in list_cat])) if is_bbox else None
        ndf_label_kpt  = np.array(convert_1d_array([x["keypoints"] if x.get("keypoints") else [] for x in list_cat])) if is_kpt else None
        transformed = self.composer.to_custom_dict(
            image=img,
            bboxes=[x["bbox"] if x.get("bbox") else [] for x in list_anns],
            mask=[x["segmentation"] if x.get("segmentation") else [] for x in list_anns],
            keypoints=[x["keypoints"] if x.get("keypoints") else [] for x in list_anns],
        )
        transformed = self.composer.preproc(transformed)
        transformed = self.composer.aftproc(transformed)
        if ndf_label_bbox is not None and len(ndf_label_bbox) > 0:
            transformed["label_bbox"] = ndf_label_bbox[transformed["label_bbox"]].tolist()
        if ndf_label_kpt is not None and len(ndf_label_kpt) > 0:
            transformed["label_kpt"] = ndf_label_kpt[transformed["label_kpt"]].tolist()
        self._show(transformed, is_bbox=is_bbox, is_mask=is_mask, is_kpt=is_kpt)

    def samples(self, item: Union[int, str], max_samples: int=10, is_bbox: bool=True, is_mask: bool=True, is_kpt: bool=True):
        img = self.get_image(item)
        list_anns, list_cat = self.coco[item]
        ndf_label_bbox = np.array(convert_1d_array([x["name"] for x in list_cat])) if is_bbox else None
        ndf_label_kpt  = np.array(convert_1d_array([x["keypoints"] if x.get("keypoints") else [] for x in list_cat])) if is_kpt else None
        for _ in range(max_samples):
            transformed = self.composer(
                image=img,
                bboxes=[x["bbox"] if x.get("bbox") else [] for x in list_anns] if is_bbox else None, 
                mask=[x["segmentation"] if x.get("segmentation") else [] for x in list_anns] if is_mask else None,
                keypoints=[x["keypoints"] if x.get("keypoints") else [] for x in list_anns]if is_kpt else None,
            )
            if ndf_label_bbox is not None and len(ndf_label_bbox) > 0:
                transformed["label_bbox"] = ndf_label_bbox[transformed["label_bbox"]].tolist()
            if ndf_label_kpt is not None and len(ndf_label_kpt) > 0:
                transformed["label_kpt"] = ndf_label_kpt[transformed["label_kpt"]].tolist()
            self._show(transformed, is_bbox=is_bbox, is_mask=is_mask, is_kpt=is_kpt)
        return transformed
