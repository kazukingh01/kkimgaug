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

### Detectron2

Version dependency.
|detectron2|kkimgaug|
|:--|:--|
|v0.3|v1.0.4|
|v0.4.1|v1.0.4|
|v0.5|v1.0.4|

If you use in detectron2, see https://github.com/kazukingh01/kkimgaug/blob/main/tests/det2_sample.py#L42. Use Mapper.

The definition is as follows.
https://github.com/kazukingh01/kkimgaug/blob/main/kkimgaug/lib/aug_det2.py#L66-L86
```
python det2_sample.py ---train
```
\# You can use official config format. (use ./config_official.json config in this code.)
```
python det2_sample.py ---train ---official
```
and you can see applied augmentaions in detectron2.
Images applied augmentaions is in "output_augmentations" directory.
```
python det2_sample.py ---prev
```
\# You can use official config format. (use ./config_official.json config in this code.)
```
python det2_sample.py ---prev ---official
```
