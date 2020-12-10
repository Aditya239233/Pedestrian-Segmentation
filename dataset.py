import os
import numpy as np
from PIL import Image
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms = None):
        self.root = root
        self.transforms = transforms

        self.images = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def __getitem__(self, idx: int):
        image_path = self.root + "PNGImages/" + self.images[idx] 
        mask_path = self.root + "PedMasks/" + self.masks[idx]

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        # Need to convert the mask to RGB because each colour corresponds to a different 
        # instance with 0 being the background
        mask = np.array(mask)
        obj_ids = np.unique(mask)[1:]

        masks = mask == obj_ids[:, None, None] # Split into Boolean masks
        # mask.shape = (num_objs, H, W)
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax]) #bounding boxes

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_idx = torch.as_tensor((idx))
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # Area of bounding box
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # Assume all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_idx"] = image_idx
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def __len__(self):
        return len(self.images)

