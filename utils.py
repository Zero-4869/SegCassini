import numpy as np
from PIL import Image
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os

def split_into_patches(x:torch.Tensor, patch_size:Union[int, Tuple[int, int]]):
    if type(patch_size).__name__ == "int":
        patch_size = (patch_size, patch_size)
    C, H, W = x.shape
    Ny = (int)(H//patch_size[0])
    Nx = (int)(W//patch_size[1])
    x_new = torch.empty((Ny * Nx, C, patch_size[0], patch_size[1]), dtype = x.dtype, device = x.device)
    for i in range(Ny):
        for j in range(Nx):
            x_new[i * Nx + j] = x[:, i * patch_size[0]: (i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]]
    return x_new

def merge_into_tile(x:List[torch.Tensor], size:Union[int, Tuple[int, int]]):
    if type(size).__name__ == "int":
        size = (size, size)    
    assert len(x) == size[0] * size[1] 
    C, H, W = x[0].shape
    tile = torch.empty((C, H * size[0], W * size[1]), dtype=x[0].dtype, device=x[0].device)
    for i in range(size[0]):
        for j in range(size[1]):
            tile[:, H*i: H*i+H, W*j: W*j+W] = x[i*size[1]+j]
    return tile

def plot_figures(x:List, save_path):
    N = len(x)
    fig, ax = plt.subplots(N, figsize = (10, 10 * N))
    for i in range(N):
        ax[i].imshow(x[i].cpu().detach().numpy().transpose(1,2,0))
    plt.savefig(save_path)

def plot_figure2(img, save_path):
    img = img.cpu().detach().numpy().transpose(1,2,0)
    img = np.clip(255 * img, 0, 255)
    Image.fromarray(img.astype(np.uint8)).save(save_path)

def plot_mask(mask, save_path):
    assert len(mask.shape) == 2, print(mask.shape)
    mask = mask.cpu().detach().numpy()
    mask = 255 * mask
    Image.fromarray(mask.astype(np.uint8)).save(save_path)

def plot_binary(x:np.ndarray, save_path):
    '''
    x: H * W * 4
    '''
    H, W, C = x.shape
    colormap = {"forest": [124, 237, 116], "hydro": [134,231,237], "road": [237, 185, 13], "town": [237, 84, 57]}
    names = ["forest", "hydro", "road", "town"]
    basename = os.path.basename(save_path).split(".")[0]
    dirname = os.path.dirname(save_path)
    for i in range(C):   
        output_image = 255 * np.ones((H, W, 3), dtype=np.uint8) 
        output_image[..., 0] = np.floor(x[..., i] * colormap[names[i]][0])
        output_image[..., 1] = np.floor(x[..., i] * colormap[names[i]][1])
        output_image[..., 2] = np.floor(x[..., i] * colormap[names[i]][2])
        Image.fromarray(output_image).save(os.path.join(dirname, basename + f"_{names[i]}.png"))

def add_noise(img, sigma):
    noise = sigma * torch.randn(img.shape, device = img.device, dtype = img.dtype, requires_grad = False)
    return img + noise

def compute_msv(kernel, u, ite = 1):
    '''Power iteration to compute the largest singular value'''
    for _ in range(ite):
        v = nn.functional.normalize(u @ kernel, p=2, dim=1).to(kernel.device)
        u = nn.functional.normalize(v @ kernel.T, p=2, dim=1).to(kernel.device)
    out = torch.sum(u @ kernel @ v.T)
    return out, u

def save_mid_result(image, path):
    Image.fromarray((255 * image.cpu().detach().numpy().transpose(1,2,0)).astype(np.uint8)).save(path)

from PIL import Image
def toBinary(inpath, outpath):
    image = np.array(Image.open(inpath))
    H, W, C = image.shape
    assert C==4
    colormap = {"road": [237, 185, 13], "hydro": [134,231,237], "town": [237, 84, 57], "forest": [124, 237, 116]}
    colormap2 = {"forest":[238, 242, 219], "hydro":[116, 238, 237], "road":[246,235,142], "town": [242, 235, 217]}
    binary_image = 255 * np.ones((H, W, 3), dtype=np.uint8)
    names = ["forest", "hydro", "road", "town"]
    for layer in [0, 3, 1]:
        color = colormap2[names[layer]]
        fg = np.where(image[..., layer] == 255)
        binary_image[fg] = color
    Image.fromarray(binary_image).save(outpath)


if __name__ == "__main__":
    import glob
    from tqdm import tqdm
    dirname = "patchdata2/epure_multilayers"
    outdir = "patchdata2/epure_multilayers_rgb_noroad"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for mode in ["train", "val"]:
        folder = os.path.join(dirname, mode)
        outfolder = os.path.join(outdir, mode)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        files = sorted(os.listdir(folder))
        for file in tqdm(files):
            path = os.path.join(folder, file)
            outpath = os.path.join(outfolder, file)
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            image_paths = sorted(glob.glob(os.path.join(path, "*.tif")))
            for image_path in image_paths:
                basename = os.path.basename(image_path)
                out_image_path = os.path.join(outpath, basename)
                toBinary(image_path, out_image_path)



