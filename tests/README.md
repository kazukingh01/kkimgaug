# Tests

```bash
python -i viewer_coco.py --config ./json/config.json --coco ./json/coco.json --dir ./src_labelme/img/
```

```bash
python -i viewer_coco.py --config ./json/config_official.json --coco ./json/coco.json --dir ./src_labelme/img/ --official
```

```bash
python -i viewer_sample.py --config ./json/config.json --img ./src_labelme/img/img_dog_cat.jpg
```

If you want to use for CocoDataset, you have to install torchvision.  
https://pytorch.org/get-started/locally/  

```bash
pip3 install torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -i dataset_coco.py --config ./json/config.json --coco ./json/coco.json --dir ./src_labelme/img/
```