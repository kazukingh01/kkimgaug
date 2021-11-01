# kkimgaug
- This package is wrapper package for albumentations( https://github.com/albumentations-team/albumentations ).
- You can define augmentation with config file like "https://github.com/kazukingh01/kkimgaug/blob/main/tests/config.json"
- You can use official config format. Check "https://albumentations.ai/docs/examples/serialization/". check option value "is_config_type_official"(https://github.com/kazukingh01/kkimgaug/blob/main/kkimgaug/lib/aug_base.py#L106)
- Augmentations and pameters are according to the albumentations. see documentations(https://albumentations.ai/docs/). 

## Installation
```
pip install git+https://github.com/kazukingh01/kkimgaug.git
```

## A simple example
code samples is in "https://github.com/kazukingh01/kkimgaug/tree/main/tests"
```
git clone https://github.com/kazukingh01/kkimgaug.git
cd kkimgaug
```

### Viewer (check augmentation)
You can see a image after augmentaions and check the applied augmentaion in the upper left corner of the image like this.
```
python viewer.py --dir ./img/ --coco ./coco.json --config ./config.json 
```
\# You can use official config format.
```
python viewer.py --dir ./img/ --coco ./coco.json --config ./config_official.json ---official
```
![image](https://i.imgur.com/2D8GxAY.png)

### How to use by Detectron2

Moving the scripts.
see: https://github.com/kazukingh01/kkdetectron2
