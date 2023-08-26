import argparse
from functools import partial
from PIL import Image

# local package
from kkimgaug.lib import BaseCompose
import kkimgaug.util.procs as P 
from kkimgaug.util.visualize import visualize
from kkimgaug.util.functions import convert_1d_array
parser = argparse.ArgumentParser()
parser.add_argument("--img",    type=str, required=True)
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

# Augment an image
composer = BaseCompose(
    config=args.config,
    preproc=[
        P.pil2nprgb, 
        P.check_coco_annotations,
        P.bbox_label_auto,
        P.mask_from_polygon_to_bool,
        P.kpt_from_coco_to_xy
    ],
    aftproc=[
        P.rgb2bgr,
        P.bbox_compute_from_mask,
        partial(P.get_applied_augmentations, draw_on_image=True),
        P.to_uint8,
    ],
)

img        = Image.open(args.img)
bboxes     = [[50,50,50,50], [200,100,100,100]]
label_bbox = ["test1", "test2"]
polygon    = [[[50,50,75,50,75,75,50,75], [75,75,100,75,100,100,75,100]], [[200,100,250,100,250,150,200,150], [250,150,300,150,300,200,250,200]]]
keypoints  = [[60,60,2.0,80,80,2.0], [225,125,2.0, 275,175,2.0]]  # coco format is x,y,v. "v" means if it is hidden.
label_kpt  = [["point1", "point2"], ["point1", "point2"]]
image      = P.pil2nprgb({"image": img})["image"]
visualize(
    image,
    bboxes=bboxes, class_names=[0,1], class_names_bk=label_bbox, 
    mask=P.mask_from_polygon_to_bool({"image": image, "mask": polygon})["mask"], 
    keypoints=P.kpt_from_coco_to_xy({"keypoints": keypoints})["keypoints"],
    class_names_kpt=convert_1d_array(label_kpt)
)
for _ in range(100):
    transformed = composer.__call__(image=img, bboxes=bboxes, mask=polygon, label_bbox=label_bbox, keypoints=keypoints, label_kpt=label_kpt)
    visualize(
        transformed["image"],
        bboxes=transformed["bboxes"],
        class_names=transformed["label_bbox"],
        class_names_bk=transformed["label_name_bbox"], 
        mask=transformed["mask"], keypoints=transformed["keypoints"],
        class_names_kpt=transformed["label_kpt"]
    )
