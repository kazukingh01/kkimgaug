import random, json
from typing import Union, List, Callable

import numpy as np
import cv2

# albumentations
import albumentations as A

# local files
import kkimgaug.lib.transforms as T
import kkimgaug.util.procs as P


__all__ = [
    "BaseCompose",
]


COMPOSE_LIST = ["OneOf", "Sequential"]


def create_compose(config: List[dict], **kwargs) -> A.Compose:
    """
    Creating compose from a list of dictionary written by strings
    Example::
        config = [
            {
                "class": "RandomCrop",
                "params": {
                    "width": 256,
                    "height": 256
                }
            }, 
            {
                "class": "OneOf",
                "params": {
                    "p": 0.8
                },
                "proc": [
                    {
                        "class": "Sequential",
                        "params": {
                            "p": 0.8
                        },
                        "proc": [
                            {
                                "class": "HorizontalFlip",
                                "params": {
                                    "p": 0.5
                                }
                            }, 
                            {
                                "class": "RandomBrightnessContrast",
                                "params": {
                                    "p": 0.2
                                }
                            }
                        ]
                    },
                    {
                        "class": "RandomBrightnessContrast",
                        "params": {
                            "p": 0.2
                        }
                    }
                ]
            }
        ]
    """
    def __loop(config: List[dict]):
        list_proc = []
        for dictwk in config:
            if dictwk["class"] in COMPOSE_LIST:
                list_proc.append(
                    getattr(A, dictwk["class"])(__loop(dictwk["proc"]), **(dictwk["params"] if dictwk["params"] else {}))
                )
            else:
                for key, val in dictwk["params"].items():
                    if isinstance(val, str) and val[:4] == "cv2.":
                        dictwk["params"][key] = getattr(cv2, val[4:])
                list_proc.append(
                    (getattr(T, dictwk["class"])(**(dictwk["params"] if dictwk["params"] else {}))) if hasattr(T, dictwk["class"]) else \
                    (getattr(A, dictwk["class"])(**(dictwk["params"] if dictwk["params"] else {}))) 
                )
        return list_proc
    return A.ReplayCompose(__loop(config), **kwargs)



class BaseCompose:
    def __init__(
        self, config: Union[str, dict], 
        preproc: List[Callable[[dict], dict]]=[
            P.bgr2rgb, 
            P.check_coco_annotations,
            P.bbox_label_auto,
            P.mask_from_polygon_to_bool, 
            P.kpt_from_coco_to_xy
        ], 
        aftproc: List[Callable[[dict], dict]]=[
            P.rgb2bgr,
            P.to_uint8,
            P.mask_inside_bbox
        ],
        is_config_type_official: bool=False,
    ):
        """
        ref: https://qiita.com/kurilab/items/b69e1be8d0224ae139ad
        Params::
            config:
                json path or dict.
                If use dict type, refer to "create_compose"
                proc_names: list of process name.
                format: string. "coco"
                [process name]: 
                    detail sequential augmentation processes.
                    proc: compose config.
                    scale: list of int. or null
                        array 0: area size.
                        array 1:
                            0: less
                            1: over
                    p: probability
            preproc, aftproc:
                callable function or list of callable functions.
                function input is dictionary type.
                    dict["image"]: np.ndarray[uint8]
                    dict["bboxes"]: [[x_min, y_min, w, h], [x_min, y_min, w, h], ...]
                    dict["mask"]: np.ndarray[bool]
                    dict["label_bbox"]: List[str]. class names.
                    dict["keypoints"]: [[x1, y1], [x2, y2], ..]
                    dict["label_kpt"]: List[str]. keypoint class names.
            is_config_type_official:
                If true, you can use official config format.
                But you can not use my original multi compose process.
                https://albumentations.ai/docs/examples/serialization/
        Example::
            config = {
                "proc_names": [
                    "proc1", "proc2"
                ],
                "format": "coco",
                "proc1": {
                    "proc": [
                        {
                            "class": "RandomCrop",
                            "params": {
                                "width": 256,
                                "height": 256
                            }
                        }
                    ],
                    "scale": [
                        100, 0
                    ],
                    "p": 1
                }, 
                "proc2": {
                    "proc": [
                        {
                            "class": "OneOf",
                            "params": {
                                "p": 0.8
                            },
                            "proc": [
                                {
                                    "class": "HorizontalFlip",
                                    "params": {
                                        "p": 0.5
                                    }
                                }, 
                                {
                                    "class": "RandomBrightnessContrast",
                                    "params": {
                                        "p": 0.2
                                    }
                                }
                            ]
                        }
                    ],
                    "scale": null,
                    "p": 0.5
                }
            }
        """
        super(BaseCompose, self).__init__()
        self.list_proc  = []
        self.list_scale = []
        self.list_p     = []
        self._preproc   = (preproc if isinstance(preproc, list) else [preproc]) if preproc else []
        self._aftproc   = (aftproc if isinstance(aftproc, list) else [aftproc]) if aftproc else []
        self.label_bbox_fields: List[str]=['label_bbox'] # Fix value
        self.label_kpt_fields : List[str]=['label_kpt' ] # Fix value
        self.box_format, self.box_params, self.keypoint_format, self.keypoint_params = None, None, None, None
        if is_config_type_official == False:
            # load my original format config
            if isinstance(config, str):
                config = json.load(open(config))
            self.box_format = config.get("bbox_params").get("format") if config.get("bbox_params") and config.get("bbox_params").get("format") else "coco"
            self.box_params = config["bbox_params"] if config.get("bbox_params") else {}
            if self.box_params.get("format"): del self.box_params["format"]
            self.keypoint_format = config.get("keypoint_params").get("format") if config.get("keypoint_params") and config.get("keypoint_params").get("format") else "xy"
            self.keypoint_params = config["keypoint_params"] if config.get("keypoint_params") else {}
            if self.keypoint_params.get("format"): del self.keypoint_params["format"]
            for proc_name in config["proc_names"]:
                self.list_proc.append(
                    create_compose(\
                        config[proc_name]["proc"],
                        bbox_params=A.BboxParams(self.box_format, **self.box_params, label_fields=self.label_bbox_fields),
                        keypoint_params=A.KeypointParams(self.keypoint_format, **self.keypoint_params, label_fields=self.label_kpt_fields)
                    )
                )
                self.list_p.append(config[proc_name]["p"])
                if config[proc_name]["scale"]:
                    self.list_scale.append(config[proc_name]["scale"])
                else:
                    self.list_scale.append([-1, -1])
        else:
            # load official format config
            self.list_proc.append(A.load(config))
            if isinstance(self.list_proc[0], A.ReplayCompose) == False:
                raise Exception(
                    f"My package work only ReplayCompose. Please change your class: {type(self.list_proc[0])} to albumentations.core.composition.ReplayCompose"
                )
            self.list_p.append(1)
            self.list_scale.append([-1, -1])
        self.list_p     = np.array(self.list_p)
        self.list_scale = np.array(self.list_scale).reshape(-1, 2)

    def __call__(
        self, image: np.ndarray, 
        bboxes=None, mask=None, label_bbox=None, keypoints=None, label_kpt=None, 
        **kwargs
    ):
        """
        Run processes to a image and annotations.
        We can decide a process in definition based on bbox's size.
        Params::
            # Input depends on "self.preproc"
            # If the preproc is below
            # preproc=[
            #     P.bbox_label_auto,
            #     P.mask_from_polygon_to_bool, 
            #     P.kpt_from_coco_to_xy
            # ]
            bboxes     = [[50,50,50,50], [200,100,100,100]]
            label_bbox = ["test1", "test2"]
            mask       = [[[50,50,75,50,75,75,50,75], [75,75,100,75,100,100,75,100]], [[200,100,250,100,250,150,200,150], [250,150,300,150,300,200,250,200]]]
            keypoints  = [[60,60,2.0,80,80,2.0], [225,125,2.0, 275,175,2.0]]  # coco format is x,y,v. "v" means if it is hidden.
            label_kpt  = [["point1", "point2"], ["point1", "point2"]]
        Output::
            >>> transformed["bboxes"]
            [[130, 90, 222, 263], [350, 111, 167, 236]]
            >>> transformed["label_bbox"]
            [0, 1]
            >>> transformed["label_name_bbox"]
            ['dog', 'cat']
            >>> transformed["mask"] # The number means label's mask.
            [[0 0 0 ... 0 0 0]
            [0 0 0 ... 0 0 0]
            [0 0 0 ... 0 0 0]
            ...
            [0 0 0 ... 0 0 0]
            [0 0 0 ... 0 0 0]
            [0 0 0 ... 0 0 0]]
            >>> (transformed["mask"] == 1).sum()
            1347
            >>> (transformed["mask"] == 2).sum()
            5193
            >>> (transformed["mask"] == 0).sum()
            213892
            >>> transformed["keypoints"]
            [(281.30315795579776, 131.97565335435658), (341.4924686880193, 147.30948383433747), (338.7031756782723, 167.9537139992281), (446.4250097260731, 130.23812203836482), (410.40681888242216, 145.69991071374542), (434.98165090003863, 138.82878180782006), (439.84763520625563, 150.04236534544637)]
            >>> transformed["label_kpt"]
            ['eye_right', 'nose', 'mouth', 'eye_left', 'eye_right', 'nose', 'mouth']        
        """
        # define dictionary
        transformed = self.to_custom_dict(image=image, bboxes=bboxes, mask=mask, label_bbox=label_bbox, keypoints=keypoints, label_kpt=label_kpt)
        # pre processing
        transformed = self.preproc(transformed)
        # choose one compose
        ndf_bool = np.ones_like(self.list_p).astype(bool)
        if bboxes is not None:
            area = float("inf")
            for _, _, width, height in bboxes:
                if area > width * height:
                    area = width * height
            ndf_bool = (
                ((self.list_scale[:, 0] <  area) & (self.list_scale[:, 1] == 0)) | \
                ((self.list_scale[:, 0] >= area) & (self.list_scale[:, 1] == 1)) | \
                (self.list_scale[:, 1] == -1)
            )
        ndf_p = self.list_p[ndf_bool]
        if ndf_p.shape[0] == 0:
            raise Exception(f'No Compose is applied.')
        ndf_p  = np.cumsum(ndf_p / ndf_p.sum())
        i_comp = np.where(ndf_p >= random.random())[0].min()
        # main processing
        transformed = self.list_proc[i_comp](**transformed)
        # after processing
        transformed = self.aftproc(transformed)
        return transformed

    def replay(
        self, info_replay, image: np.ndarray, 
        bboxes=None, mask=None, label_bbox=None, keypoints=None, label_kpt=None, 
        is_preproc: bool=True, is_aftproc: bool=True,
        **kwargs
    ):
        """
        Usage::
            self.replay(transformed['replay'], image, ...)
        """
        # define dictionary
        transformed = self.to_custom_dict(image=image, bboxes=bboxes, mask=mask, label_bbox=label_bbox, keypoints=keypoints, label_kpt=label_kpt)
        # pre processing
        if is_preproc:
            transformed = self.preproc(transformed)
        # main processing. 0 index fix.
        transformed = self.list_proc[0].replay(info_replay, **transformed)
        # after processing
        if is_aftproc:
            transformed = self.aftproc(transformed)
        return transformed

    @classmethod
    def to_custom_dict(cls, image: np.ndarray, bboxes=None, mask=None, label_bbox=None, keypoints=None, label_kpt=None, **kwargs):
        # image copy
        image = image.copy()
        # define dictionary
        transformed = {}
        transformed["image"]  = image
        transformed["bboxes"] = bboxes if bboxes is not None else []
        if mask is not None and len(mask) > 0: transformed["mask"] = mask
        transformed["label_bbox"] = label_bbox if label_bbox is not None else []
        transformed["keypoints"]  = keypoints if keypoints is not None else []
        transformed["label_kpt"]  = label_kpt if label_kpt is not None else []
        for x, y in kwargs.items():
            transformed[x] = y
        return transformed

    def preproc(self, transformed: dict):
        for _proc in self._preproc:
            transformed = _proc(transformed)
        return transformed

    def aftproc(self, transformed: dict):
        for _proc in self._aftproc:
            transformed = _proc(transformed)
        return transformed