# MobileNet v2 PyTroch on Food101
Training of MobileNet v2 from Torchvision on Food101 dataset

## Run training:
```
python train.py --epochs 30 -j 6 -b 256 --lr 0.001 --pretrained --wd 10e-5 dataset
```

## Export to ONNX:
```
python export.py model_best.pth.tar
```
