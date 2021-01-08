import albumentations as A
import os
import cv2
import json

# local package
from kkimgaug.util.functions import get_args
from kkimgaug.lib import BaseCompose, CocoVisualizer

args = get_args()

# Augment an image
visualizer = CocoVisualizer(
    config=args.get("config"),
    coco_json=args.get("coco"),
    image_dir=args.get("dir")
)
visualizer.show(0)
transformed = visualizer.samples(0 if args.get("img") is None else os.path.basename(args.get("img")), 100, is_kpt=True, is_mask=True)
