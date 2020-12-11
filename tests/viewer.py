import albumentations as A
import cv2
import json

# local package
from kkimgaug.lib import BaseCompose, CocoVisualizer

image = cv2.imread("./img/img_dog_cat.jpg")

# Augment an image
visualizer = CocoVisualizer(
    config="./config.json",
    coco_json="./coco.json",
    image_dir="./img"
)
visualizer.show(0)
transformed = visualizer.samples(0, 100, is_kpt=False, is_mask=False)
