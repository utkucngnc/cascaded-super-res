from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import torch as th

def getConfig(path: str):
    return OmegaConf.load(path)

def showImg(img: th.Tensor):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()