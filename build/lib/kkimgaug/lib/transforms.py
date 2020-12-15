import random
import cv2
import albumentations as A
import albumentations.augmentations.functional as F


__all__ = [
    "MyRandomSizedCrop",
]


class MyRandomSizedCrop(A.DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(self, min_max_h_scale: (float, float), min_max_w_scale: (float, float), always_apply=False, p=1.0):
        super(MyRandomSizedCrop, self).__init__(always_apply, p)
        self.min_max_h_scale = min_max_h_scale
        self.min_max_w_scale = min_max_w_scale
        self._height = 0
        self._width  = 0

    def apply(self, img, h_start=0, w_start=0, scale_h=0, scale_w=0, **params):
        self._height = int(img.shape[0] * scale_h)
        self._width  = int(img.shape[1] * scale_w)
        return F.random_crop(img, self._height, self._width, h_start, w_start)

    def get_params(self):
        scale_h = self.min_max_h_scale[0] + (self.min_max_h_scale[1] - self.min_max_h_scale[0]) * random.random()
        scale_w = self.min_max_w_scale[0] + (self.min_max_w_scale[1] - self.min_max_w_scale[0]) * random.random()
        return {"h_start": random.random(), "w_start": random.random(), "scale_h": scale_h, "scale_w": scale_w}

    def apply_to_bbox(self, bbox, scale_h=0, scale_w=0, **params):
        return F.bbox_random_crop(bbox, self._height, self._width, **params)

    def apply_to_keypoint(self, keypoint, scale_h=0, scale_w=0, **params):
        return F.keypoint_random_crop(keypoint, self._height, self._width, **params)

    def get_transform_init_args_names(self):
        return ("min_max_h_scale", "min_max_w_scale",)
