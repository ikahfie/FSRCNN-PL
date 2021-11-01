# FSRCNN-PL
FSRCNN with VGG16 Perceptual Loss.
---
Faster Super Resolution CNN  (Dong, *et al*., 2016) modified to
use VGG16 perceptual loss.
Implemented using Tensorflow 2.6. Trained using 200 images from DIV2K_TRAIN_HR dataset

## Training Guide
---
1. Open ```run_train.py```
2. Edit ```config``` dictionary variables.
3. Edit ```image_config.py``` inside ```utils/```
4. Run ```python run_train.py```

## Dependacies
---
* ```Tensorflow```
* ```PIL```
* ```Albumentations```
* ```numpy```

## TO-DO LIST
---
- [x] implement FSRCNN 
- [x] implement preprocess 
- [x] implement training routine 
- [x] implement vgg-loss 
- [ ] implement test prediction routine