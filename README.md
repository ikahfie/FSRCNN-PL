# FSRCNN-PL
FSRCNN with VGG16 Perceptual Loss.
---
Faster Super Resolution CNN  (Dong, *et al*., 2016) modified to
use VGG16 perceptual loss.
Implemented using Tensorflow 2.6. Trained using images from DIV2K_TRAIN_HR dataset

## Training Guide
---
1. Download a pretrained VGG16 hdf5 model and put it either in ./ or utils/
2. Open ```run_train.py```
3. Edit ```config``` dictionary variables.
4. Edit variables inside of ```image_config.py```
5. Run ```python run_train.py```

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