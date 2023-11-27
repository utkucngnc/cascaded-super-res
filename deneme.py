from dataset import CustomDataset as CD
from utils import *
import torchvision.transforms as transforms

def main():
    cfg = getConfig('./conf.yaml')
    transform = transforms.Compose([transforms.Resize(cfg.dataset.image.shape, antialias=True),
                                    transforms.Grayscale(num_output_channels=cfg.dataset.image.num_channels),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor()])
    
    ds = CD(cfg.dataset.path, transform=transform)

    print(ds[0].dtype)

if __name__ == "__main__":
    main()