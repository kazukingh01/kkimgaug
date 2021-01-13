import albumentations as A
import os
import cv2
import json

# local package
from kkimgaug.util.functions import get_args
from kkimgaug.lib import BaseCompose, Visualizer

args = get_args()

# Augment an image
visualizer = Visualizer(
    config=args.get("config"),
    coco_json=args.get("coco"),
    image_dir=args.get("dir")
)
img = 0
if isinstance(args.get("img"), str):
    if args.get("img").isnumeric():
        img = int(args.get("img"))
    else:
        img = os.path.basename(args.get("img"))
visualizer.show(img)
transformed = visualizer.show(img, is_aug=True, max_samples=100, is_bbox=True, is_kpt=True, is_mask=True)
