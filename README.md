# kkimgaug

## Installation
```
pip install https://github.com/kazukingh01/kkimgaug.git
```

## A simple example
samples is in "https://github.com/kazukingh01/kkimgaug/tree/main/tests"
* config.json = "https://github.com/kazukingh01/kkimgaug/blob/main/tests/config.json"
* coco.json   = "https://github.com/kazukingh01/kkimgaug/blob/main/tests/coco.json"
* image_dir   = "https://github.com/kazukingh01/kkimgaug/tree/main/tests/img"
```
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
visualizer.samples(0, 100)
```
