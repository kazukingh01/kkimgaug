import argparse
# local package
from kkimgaug.lib import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument("--config",  type=str, required=True)
parser.add_argument("--coco",    type=str, required=True)
parser.add_argument("--dir",     type=str, required=True)
parser.add_argument("--official",action='store_true', default=False)
args = parser.parse_args()

# Augment an image
visualizer = Visualizer(
    config=args.config,
    coco_json=args.coco,
    image_dir=args.dir,
    is_config_type_official=args.official
)
img = 0
visualizer.show(img)
transformed = visualizer.show(
    img, is_aug=True, max_samples=100, 
    is_bbox=True, is_kpt=True, is_mask=True, resize=256
)
