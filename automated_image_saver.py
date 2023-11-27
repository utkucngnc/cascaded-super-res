from skimage import io
from skimage import transform
import os
from tqdm import tqdm
import numpy as np

def save_imgs(load_path: str, save_path: str, ext: str, resize: tuple = None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i,img in enumerate(tqdm(io.imread(load_path).astype(np.uint8))):
        if resize is None:    
            io.imsave(save_path + str(i) + ext, img)
        else:
            io.imsave(save_path + str(i) + ext, transform.resize(img, resize, anti_aliasing=True))

if __name__ == '__main__':
    load_path = '../Pristine/PTY_pristine_raw.tif'
    save_path = './data/'
    ext = '.png'
    save_imgs(load_path, save_path, ext, resize=(64,64))
    print('Done.')