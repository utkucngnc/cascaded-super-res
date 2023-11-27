import numpy as np
import skimage.io as io
from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms
import torch as th
import os

image_formats = ['jpg', 'jpeg', 'png', 'tif', 'bmap']

class CustomDataset(Dataset):
    def __init__(self, path: str, transform: None) -> None:
        super().__init__()
        self.path = path
        self.transform = transform
        self.data = None
        self.tif = False

        if not os.path.exists(self.path):
            raise Exception("Path does not exist!")
        
        if self.path.endswith(".tif"):
            self.data = io.imread(self.path)
            self.tif = True
        else:
            num_file = 0
            for _, file in enumerate(os.listdir(self.path)):
                if self.check_image(file):
                    num_file += 1
                    self.data[num_file] = file

    def __len__(self):
        return len(self.data)
    
    #   Returns a single image, type: np.ndarray, dtype: np.uint8
    def __getitem__(self, index):
        if self.tif:
            image = self.data[index]
        else:
            image = io.imread(self.data[index])
        
        return self.conv_to_tensor(image)
    
    def check_image(self, image: str):
        if image.split('.')[-1] not in image_formats:
            raise Exception("Image format not supported!")
        else:
            return True
    
    def conv_to_tensor(self, image: np.ndarray):
        image = Image.fromarray(image)
        if self.transform:
            return self.transform(image)
        return th.from_numpy(image).unsqueeze(0).type(th.float32)