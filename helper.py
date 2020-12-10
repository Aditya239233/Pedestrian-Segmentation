import torchvision
from vision.references.detection import utils
import vision.references.detection.transforms as T

def get_transform(train):
    # data augmentation
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

def add_pallet(mask):
    mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
    ])
    return mask

def collate_fn(batch):
    return tuple(zip(*batch))