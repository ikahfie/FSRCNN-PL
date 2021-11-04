from PIL import Image

IMAGE_FORMAT = ".png"
DOWNSAMPLE_MODE = Image.BICUBIC
COLOR_CHANNELS = 3

UPSCALING_FACTOR = 2
HR_IMG_SIZE = (512, 512) #size is selected beased on the smallest image in the dataset
LR_IMG_SIZE = (HR_IMG_SIZE[0] // UPSCALING_FACTOR, HR_IMG_SIZE[1] // UPSCALING_FACTOR)

PADDING = 0 
INIT_TILE_SIZE = (64, 64) # minimum lr_tile size is 16x16 for VGG LOSS



