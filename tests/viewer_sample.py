import argparse
import numpy as np
from functools import partial
from PIL import Image

# local package
from kkimgaug.lib import BaseCompose
import kkimgaug.util.procs as P 
from kkimgaug.util.visualize import visualize
from kkimgaug.config.config import LABEL_NAME_IMAGE, LABEL_NAME_BBOX, LABEL_NAME_BBOX_CLASS, LABEL_NAME_MASK, LABEL_NAME_KPT, LABEL_NAME_KPT_CLASS


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
        P.kpt_label_auto,
        P.mask_from_polygon_to_bool,
        P.kpt_from_coco_to_xy
    ],
    aftproc=[
        P.rgb2bgr,
        P.mask_inside_bbox,
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
    bboxes=bboxes,
    class_names=[1,2],
    class_names_saved=label_bbox, 
    mask=P.mask_from_polygon_to_bool({"image": image, "mask": polygon})["mask"], 
    keypoints=P.kpt_from_coco_to_xy({"keypoints": keypoints})["keypoints"],
    class_names_kpt=[0,1,2,3],
    class_names_kpt_saved=label_kpt,
)

for ndf in np.random.randint(0, 3, (100, 5)):
    transformed = {}
    transformed[LABEL_NAME_IMAGE]      = img
    transformed[LABEL_NAME_BBOX]       = [None, [], bboxes    ][ndf[0]]
    transformed[LABEL_NAME_BBOX_CLASS] = [None, [], label_bbox][ndf[1]]
    transformed[LABEL_NAME_MASK]       = [None, [], polygon   ][ndf[2]]
    transformed[LABEL_NAME_KPT]        = [None, [], keypoints ][ndf[3]]
    transformed[LABEL_NAME_KPT_CLASS]  = [None, [], label_kpt ][ndf[4]]
    print({x:transformed.get(x) for x in [LABEL_NAME_IMAGE, LABEL_NAME_BBOX, LABEL_NAME_BBOX_CLASS, LABEL_NAME_MASK, LABEL_NAME_KPT, LABEL_NAME_KPT_CLASS]})
    transformed = composer.__call__(**transformed)
    print({x:transformed.get(x) for x in [LABEL_NAME_IMAGE, LABEL_NAME_BBOX, LABEL_NAME_BBOX_CLASS, LABEL_NAME_MASK, LABEL_NAME_KPT, LABEL_NAME_KPT_CLASS]})
    visualize(
        transformed.get(LABEL_NAME_IMAGE),
        bboxes=transformed.get(LABEL_NAME_BBOX),
        class_names=transformed.get(LABEL_NAME_BBOX_CLASS),
        class_names_saved=transformed.get(f"{LABEL_NAME_BBOX_CLASS}_saved"), 
        mask=transformed.get(LABEL_NAME_MASK), keypoints=transformed.get(LABEL_NAME_KPT),
        class_names_kpt=transformed.get(LABEL_NAME_KPT_CLASS),
        class_names_kpt_saved=transformed.get(f"{LABEL_NAME_KPT_CLASS}_saved")
    )
