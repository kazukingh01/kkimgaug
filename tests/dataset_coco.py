import torchvision, argparse
from typing import List
# local package
from kkimgaug.lib import BaseCompose
import kkimgaug.util.procs as P 
from kkimgaug.util.visualize import visualize
from kkimgaug.config.config import LABEL_NAME_IMAGE, LABEL_NAME_BBOX, LABEL_NAME_BBOX_CLASS, LABEL_NAME_MASK, LABEL_NAME_KPT, LABEL_NAME_KPT_CLASS


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--coco",   type=str, required=True)
parser.add_argument("--dir",    type=str, required=True)
args = parser.parse_args()


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, root: str, annFile: str, transform_config: str=None):
        super().__init__(root, annFile)
        self.dict_cat_id   = {y["name"]: x for x, y in self.coco.cats.items()}
        self.transform_alb = BaseCompose(
            config=transform_config,
            preproc=[
                P.pil2nprgb, 
                P.check_coco_annotations,
                P.bbox_label_auto,
                P.kpt_label_auto,
                P.mask_from_polygon_to_bool,
                P.kpt_from_coco_to_xy
            ],
            aftproc=[
                P.rgb2bgr,
                P.mask_inside_bbox,
                P.bbox_compute_from_mask,
                P.mask_from_bool_to_polygon,
                P.kpt_from_xy_to_coco,
                P.to_uint8,
                P.npbgr2pil,
            ],
        ) if isinstance(transform_config, str) else None
    def __getitem__(self, idx, is_aug: bool=True):
        img, anno   = super().__getitem__(idx)
        if self.transform_alb is not None and is_aug:
            transformed = self.convert_from_torchvision_to_alb(img, anno)
            transformed = self.transform_alb(**transformed)
            img, anno   = self.convert_from_alb_to_torchvision(transformed, anno)
        return img, anno
    def convert_from_torchvision_to_alb(self, img, anno):
        transformed               = {}
        transformed["image"]      = img
        transformed["bboxes"]     = self.get_from_list_of_dict(anno, "bbox")
        transformed["label_bbox"] = [self.coco.cats[x]["name"] for x in self.get_from_list_of_dict(anno, "category_id")]
        transformed["mask"]       = self.get_from_list_of_dict(anno, "segmentation")
        transformed["keypoints"]  = self.get_from_list_of_dict(anno, "keypoints")
        transformed["label_kpt"]  = [self.coco.cats[x]["keypoints"] for x in self.get_from_list_of_dict(anno, "category_id")]
        return transformed
    def convert_from_alb_to_torchvision(self, transformed, anno_org):
        img  = transformed["image"]
        anno = []
        for i, (bbox, i_bbox) in enumerate(zip(transformed["bboxes"], transformed["label_bbox"])):
            i_bbox    = i_bbox - 1 # i_bbox starts from 1 not 0. 0 means background
            dict_anno = {}
            dict_anno["id"]          = i
            dict_anno["bbox"]        = bbox
            dict_anno["area"]        = int(bbox[-2] * bbox[-1])
            dict_anno["category_id"] = anno_org[i_bbox]["category_id"]
            dict_anno["image_id"]    = anno_org[i_bbox]["image_id"]
            dict_anno["iscrowd"]     = anno_org[i_bbox]["iscrowd"]
            if transformed.get("keypoints") is not None and anno_org[i_bbox].get("keypoints") is not None:
                dict_anno["keypoints"]     = transformed["keypoints"][i]
                dict_anno["num_keypoints"] = (len(transformed["keypoints"][i]) // 3)
            if transformed.get("mask") is not None and anno_org[i_bbox].get("segmentation") is not None:
                dict_anno["segmentation"] = transformed["mask"][i]
            anno.append(dict_anno)
        return img, anno
    def show(self, idx, is_aug: bool=False):
        img, anno = self.__getitem__(idx, is_aug=is_aug)
        transformed = self.convert_from_torchvision_to_alb(img, anno)
        transformed = self.transform_alb.preproc(transformed)
        visualize(
            transformed.get(LABEL_NAME_IMAGE),
            bboxes               =transformed.get(LABEL_NAME_BBOX),
            class_names          =transformed.get(LABEL_NAME_BBOX_CLASS),
            class_names_saved    =transformed.get(f"{LABEL_NAME_BBOX_CLASS}_saved"), 
            mask                 =transformed.get(LABEL_NAME_MASK),
            keypoints            =transformed.get(LABEL_NAME_KPT),
            class_names_kpt      =transformed.get(LABEL_NAME_KPT_CLASS),
            class_names_kpt_saved=transformed.get(f"{LABEL_NAME_KPT_CLASS}_saved")
        )
    @classmethod
    def get_from_list_of_dict(cls, anno: List[dict], key: str) -> List[object]:
        ret = [tmp.get(key) for tmp in anno]
        if sum([x is None for x in ret]) > 0: ret = []
        return ret


dataset_alb = CocoDataset(args.dir, args.coco, transform_config=args.config)
dataset_alb.show(0, is_aug=False)
for _ in range(100):
    dataset_alb.show(0, is_aug=True)
