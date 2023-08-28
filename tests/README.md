# Tests

```bash
python -i viewer_coco.py --config config.json --coco coco.json --dir img/
```

```bash
python -i viewer_coco.py --config config_official.json --coco coco.json --dir img/ --official
```

```bash
python -i viewer_sample.py --config config.json --img ./img/img_dog_cat.jpg
```

If you want to use for CocoDataset, you have to install torchvision.  
https://pytorch.org/get-started/locally/  

```bash
pip3 install torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
python -i dataset_coco.py --config config.json --coco coco.json --dir img/
```