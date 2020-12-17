# kkimgaug
- This package is wrapper package for albumentations( https://github.com/albumentations-team/albumentations ).
- You can define augmentation with config file like "https://github.com/kazukingh01/kkimgaug/blob/main/tests/config.json"
- Augmentations and pameters are according to the albumentations. see documentations(https://albumentations.ai/docs/). 

## Installation
```
pip install git+https://github.com/kazukingh01/kkimgaug.git@v1.0.1
```

## A simple example
code samples is in "https://github.com/kazukingh01/kkimgaug/tree/main/tests"
```
git clone https://github.com/kazukingh01/kkimgaug.git
cd kkimgaug
```

### Viewer (check augmentation)
You can see the applied augmentaion in the upper left corner of the image like this.
![image](https://i.imgur.com/2D8GxAY.png)
```
python viewer.py --dir ./img/ --coco ./coco.json --config ./config.json 
```


### Detectron2( over virsion 0.3 !!)
If you use in detectron2, see https://github.com/kazukingh01/kkimgaug/blob/main/tests/det2_sample.py#L42. Use Mapper.

The definition is as follows.
https://github.com/kazukingh01/kkimgaug/blob/main/kkimgaug/lib/aug_det2.py#L64-L84
```
python det2_sample.py ---train
```
and you can see applied augmentaions in detectron2.
Images applied augmentaions is in "output_augmentations" directory.
```
python det2_sample.py ---prev
```
