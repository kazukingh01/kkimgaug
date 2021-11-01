import os
# local package
from kkimgaug.util.functions import get_args
from kkimgaug.lib import Visualizer

args = get_args()

# Augment an image
visualizer = Visualizer(
    config=args.get("config"),
    coco_json=args.get("coco"),
    image_dir=args.get("dir"),
    is_config_type_official=True if args.get("official") is not None else False
)
img = 0
if isinstance(args.get("img"), str):
    if args.get("img").isnumeric():
        img = int(args.get("img"))
    else:
        img = os.path.basename(args.get("img"))
visualizer.show(img)
transformed = visualizer.show(img, is_aug=True, max_samples=100, is_bbox=True, is_kpt=True, is_mask=True, resize=1000)
