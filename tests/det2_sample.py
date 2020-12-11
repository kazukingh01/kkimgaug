import os, copy
import cv2
import numpy as np
import torch

# local packages
from kkimgaug.util.functions import convert_polygon_to_bool, correct_dirpath, get_args

# detectron2 packages
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T



class Det2Simple(DefaultTrainer):
    """
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=ZyAvNCJMmvFF
    公式の tutorial を参考にして作成した最も simple な class. debug で使用する
    """
    def __init__(self, dataset_name: str=None, coco_json_path: str=None, image_root: str=None, n_classes: int=1, aug_config: str=None):
        self.dataset_name = dataset_name
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
        self.cfg = cfg
        self.mapper = None
        if self.dataset_name is None: raise Exception("dataset name is None !")
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        cfg.DATASETS.TRAIN = (self.dataset_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
        cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes  # only has one class (ballon)
        register_coco_instances(self.dataset_name, {}, coco_json_path, image_root)
        self.mapper = Mapper(self.cfg, True, config=aug_config) if aug_config is not None else None
        super().__init__(self.cfg)
        self.resume_or_load(resume=False)
    
    def set_predictor(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2   # set the testing threshold for this model
        if self.dataset_name is not None:
            self.cfg.DATASETS.TEST = (self.dataset_name, )
        self.predictor = DefaultPredictor(self.cfg)

    def show(self, file_path: str):
        if not hasattr(self, "predictor"):
            self.set_predictor()
        img = cv2.imread(file_path)
        data = self.predictor(img)
        img = self.draw_annoetation(img, data)
        cv2.imshow("test", img)
        cv2.waitKey(0)

    def draw_annoetation(self, img: np.ndarray, data: dict):
        metadata = MetadataCatalog.get(self.dataset_name)
        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata, 
            scale=1.0, 
            instance_mode=ColorMode.IMAGE #ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(data["instances"].to("cpu"))
        img_ret = v.get_image()[:, :, ::-1]
        return img_ret

    def build_train_loader(self, cfg):
        return build_detection_train_loader(cfg, mapper=self.mapper)

    @classmethod
    def img_from_tensor(cls, data):
        img = data["image"].detach().numpy().copy().T.astype(np.uint8)
        img = np.rot90(img, 1)
        img = np.flipud(img)
        return img

    def preview_augmentation(self, outdir: str="./output_augmentations", n_output: int=100):
        outdir = correct_dirpath(outdir)
        os.makedirs(outdir, exist_ok=True)
        count = 0
        for i, x in enumerate(self.data_loader):
            # x には per batch 分の size (2個とか) 入っているので、それ分回す
            for j, data in enumerate(x):
                if j > 0: continue
                ## Visualizer を predictor と統一するため, gt_*** -> pred_*** に copy する
                img = self.img_from_tensor(data)
                ins = data["instances"].to("cpu")
                if ins.has("gt_boxes"):     ins.set("pred_boxes",     ins.gt_boxes)
                if ins.has("gt_classes"):   ins.set("pred_classes",   ins.gt_classes)
                if ins.has("gt_keypoints"): ins.set("pred_keypoints", ins.gt_keypoints)
                if ins.has("gt_masks"):
                    ## gt_mask では [x1, y1, x2, y2, ... ]の形式になっているのでそれを pred [False, True, True, ...] 形式に変換する
                    segs = ins.get("gt_masks").polygons
                    list_ndf = []
                    for seg_a_class in segs:
                        ndf = convert_polygon_to_bool(img.shape[0], img.shape[1], seg_a_class)
                        list_ndf.append(ndf)
                    ndf = np.concatenate([[ndfwk] for ndfwk in list_ndf], axis=0)
                    ins.set("pred_masks", torch.from_numpy(ndf))  # Tensor 形式に変換
                data["instances"] = ins
                img = self.draw_annoetation(img, data)
                cv2.imwrite(outdir + "preview_augmentation." + str(i) + "." + str(j) + ".png", img)
            count += 1
            if count > n_output: break

from kkimgaug.lib.aug_det2 import Det2Compose
class Mapper(DatasetMapper):
    """
    DatasetMapper 内の Detectron2 Augmentation が格納されている場所
    det2._trainer.data_loader.dataset.dataset._map_func._obj.augmentations.augs
    """
    def __init__(self, *args, config: str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.composer = Det2Compose(config)

    def __call__(self, dataset_dict):
        """
        copy "__call__" function of detectron2.data.dataset_mapper.DatasetMapper 
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        ##### My Code #####
        transformed = self.composer(
            image=image,
            bboxes=[x["bbox"] if x.get("bbox") else [] for x in dataset_dict["annotations"]],
            label_bbox=[x["category_id"] if x.get("category_id") is not None else "" for x in dataset_dict["annotations"]],
            mask=[x["segmentation"] if x.get("segmentation") else [] for x in dataset_dict["annotations"]],
            keypoints=[x["keypoints"] if x.get("keypoints") else [] for x in dataset_dict["annotations"]],
        )
        image = transformed["image"]
        if "annotations" in dataset_dict:
            for i, dictwk in enumerate(dataset_dict["annotations"]):
                if "bbox" in dictwk:
                    dictwk["bbox"] = transformed["bboxes"][i]
                if "keypoints" in dictwk:
                    dictwk["keypoints"] = transformed["keypoints"][i]
                if "segmentation" in dictwk:
                    dictwk["segmentation"] = transformed["mask"][i]
        ##### My Code #####

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


if __name__ == "__main__":
    args = get_args()
    det2 = Det2Simple(
        dataset_name="test",
        coco_json_path="./coco.json",
        image_root="./img",
        aug_config="./config.json",
        n_classes=2
    )
    image = cv2.imread("./img/img_dog_cat.jpg")
    if   args.get("prev") is not None:
        det2.preview_augmentation()
    elif args.get("train") is not None:
        det2.train()
