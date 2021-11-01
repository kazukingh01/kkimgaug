import os, copy
import cv2
import numpy as np
import torch

# local packages
from kkimgaug.lib.aug_det2 import Mapper
from kkimgaug.util.functions import convert_polygon_to_bool, correct_dirpath, get_args

# detectron2 packages
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog


class Det2Simple(DefaultTrainer):
    """
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=ZyAvNCJMmvFF
    公式の tutorial を参考にして作成した最も simple な class. debug で使用する
    """
    def __init__(self, dataset_name: str=None, coco_json_path: str=None, image_root: str=None, n_classes: int=1, aug_config: str=None, is_config_type_official: bool=False):
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
        self.mapper = Mapper(self.cfg, True, config=aug_config, is_config_type_official=is_config_type_official) if aug_config is not None else None
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


if __name__ == "__main__":
    args = get_args()
    if args.get("official") is None:
        det2 = Det2Simple(
            dataset_name="test",
            coco_json_path="./coco.json",
            image_root="./img",
            aug_config="./config.json",
            n_classes=2,
            is_config_type_official=False
        )
    else:
        # If you want to use official json format, like this
        det2 = Det2Simple(
            dataset_name="test",
            coco_json_path="./coco.json",
            image_root="./img",
            aug_config="./config_official.json",
            n_classes=2,
            is_config_type_official=True
        )
    image = cv2.imread("./img/img_dog_cat.jpg")
    if   args.get("prev") is not None:
        det2.preview_augmentation()
    elif args.get("train") is not None:
        det2.train()
    if args.get("pred") is not None:
        det2.show("./img/img_dog_cat.jpg")