from typing import Union
from functools import partial
import numpy as np

# local files
from kkimgaug.lib import BaseCompose
from kkimgaug.util.procs import bgr2rgb, rgb2bgr, mask_from_polygon_to_bool, mask_from_bool_to_polygon, kpt_from_coco_to_xy, to_uint8, restore_kpt_coco_format, get_applied_augmentations


class Det2Compose(BaseCompose):
    def __init__(
        self, config: Union[str, dict], 
        drow_on_image: bool=False
    ):
        super().__init__(
            config=config,
            preproc=[
                bgr2rgb, 
                mask_from_polygon_to_bool,
                kpt_from_coco_to_xy
            ],
            aftproc=[
                rgb2bgr, 
                mask_from_bool_to_polygon,
                restore_kpt_coco_format,
                partial(get_applied_augmentations, drow_on_image=drow_on_image),
                to_uint8,
            ]
        )
