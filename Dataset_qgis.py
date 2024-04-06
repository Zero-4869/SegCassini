from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np
from utils import split_into_patches
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision import transforms
import torch


class CycleGANDataset_train_qgis(Dataset):
    '''store the url, output in 0-1 because of ToTensor
        dir1: directory for Cassini maps
        dir2: directory for vector layers
        mask_dir1: mask for Cassini maps
    '''
    def __init__(self, dir1, dir2, mask_dir1, patch_size, transform = None):
        super(CycleGANDataset_train_qgis, self).__init__()
        dirs = [dir1, dir2, mask_dir1]
        _paths = []
        for _dir in dirs:
            _path = []
            for _file in sorted(os.listdir(_dir)):
                _path += sorted(glob.glob(os.path.join(_dir, _file, "*.tif")))
            _paths.append(_path)
        assert len(_paths[0]) == len(_paths[1])
        assert len(_paths[0]) == len(_paths[2])
        self.paths = list(zip(*_paths))
        self.transform = transform
        self.patch_size = patch_size
        
        print(f"Totally {len(self.paths)} patches for training")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        sh = np.random.randint(0, 256 - self.patch_size)
        sw = np.random.randint(0, 256 - self.patch_size)
        imgs = [Image.open(p) for p in self.paths[item]] 
        crop_imgs = [img.crop((sw, sh, sw+self.patch_size, sh+self.patch_size)) for img in imgs]
        if self.transform:
            crop_imgs = [self.transform(img) for img in crop_imgs]
        return (crop_imgs[0], crop_imgs[1], crop_imgs[2])

class CycleGANDataset_val_qgis(Dataset):
    '''store the url, output in 0-1 because of ToTensor, for val'''
    def __init__(self, dir1, dir2, mask_dir1, transform = None):
        super(CycleGANDataset_val_qgis, self).__init__()
        dirs = [dir1, dir2, mask_dir1]
        _paths = []
        for _dir in dirs:
            _path = []
            for _file in sorted(os.listdir(_dir)):
                _path += sorted(glob.glob(os.path.join(_dir, _file, "*.tif")))
            _paths.append(_path)
        assert len(_paths[0]) == len(_paths[1])
        assert len(_paths[0]) == len(_paths[2])
        self.paths = list(zip(*_paths))
        self.transform = transform
        
        print(f"Totally {len(self.paths)} patches for training")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        imgs = [Image.open(p) for p in self.paths[item]] 
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        return (imgs[0], imgs[1], imgs[2])

class CycleGANDataset_Test_qgis(Dataset):
    '''store the url, output in 0-1 because of ToTensor, for test'''
    def __init__(self, dir1, dir2, mask_dir1, transform = None):
        super(CycleGANDataset_Test_qgis, self).__init__()
        dirs = [dir1, dir2, mask_dir1]
        _paths = []
        self.city_name = []
        for _dir in dirs:
            _path = []
            for _file in sorted(os.listdir(_dir)):
                _path += sorted(glob.glob(os.path.join(_dir, _file, "*.tif")))
                if _dir == dir1:
                    self.city_name.extend([_file]*len(glob.glob(os.path.join(_dir, _file, "*.tif"))))
            _paths.append(_path)

        assert len(_paths[0]) == len(_paths[1])
        assert len(_paths[0]) == len(_paths[2])
        assert len(_paths[0]) == len(self.city_name)
        self.paths = list(zip(*_paths))
        self.transform = transform
        
        print(f"Totally {len(self.paths)} patches for training")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        imgs = [Image.open(p) for p in self.paths[item]] 
        H, W, _ = np.array(imgs[0]).shape
        H, W = (H//64)*64, (W//64)*64
        Crop = transforms.CenterCrop((H, W))
        crop_imgs = [Crop(img) for img in imgs]
        if self.transform:
            crop_imgs = [self.transform(img) for img in crop_imgs]
        basename = os.path.basename(self.paths[item][0])
        return (crop_imgs[0], crop_imgs[1], crop_imgs[2], self.city_name[item], basename)

class SegDataset_Train(Dataset):
    '''store the url, output in 0-1 because of ToTensor
        dir1: directory for IGN maps
        dir2: directory for GT layers
    '''
    def __init__(self, dir1, dir2,  patch_size, transform = None):
        super(SegDataset_Train, self).__init__()
        dirs = [dir1, dir2]
        _paths = []
        for _dir in dirs:
            _path = []
            for _file in sorted(os.listdir(_dir)):
                _path += sorted(glob.glob(os.path.join(_dir, _file, "*.tif")))
            _paths.append(_path)
        assert len(_paths[0]) == len(_paths[1])
        self.paths = list(zip(*_paths))
        self.transform = transform
        self.patch_size = patch_size
        
        print(f"Totally {len(self.paths)} patches for training")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        sh = np.random.randint(0, 256 - self.patch_size)
        sw = np.random.randint(0, 256 - self.patch_size)
        imgs = [Image.open(p) for p in self.paths[item]] 
        crop_imgs = [img.crop((sw, sh, sw+self.patch_size, sh+self.patch_size)) for img in imgs]
        if self.transform:
            crop_imgs[0] = self.transform(crop_imgs[0])
        crop_imgs[1] = torch.tensor(np.array(crop_imgs[1]))
        return (crop_imgs[0], crop_imgs[1])

class SegDataset_Val(Dataset):
    '''store the url, output in 0-1 because of ToTensor, for val'''
    def __init__(self, dir1, dir2, transform = None):
        super(SegDataset_Val, self).__init__()
        dirs = [dir1, dir2]
        _paths = []
        for _dir in dirs:
            _path = []
            for _file in sorted(os.listdir(_dir)):
                _path += sorted(glob.glob(os.path.join(_dir, _file, "*.tif")))
            _paths.append(_path)
        assert len(_paths[0]) == len(_paths[1])
        self.paths = list(zip(*_paths))
        self.transform = transform
        
        print(f"Totally {len(self.paths)} patches for Validating")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        imgs = [Image.open(p) for p in self.paths[item]] 
        if self.transform:
            imgs[0] = self.transform(imgs[0])
        imgs[1] = torch.tensor(np.array(imgs[1]))
        return (imgs[0], imgs[1])

class SegDataset_Test(Dataset):
    '''store the url, output in 0-1 because of ToTensor, for test, without gt'''
    def __init__(self, dir, transform = None):
        super(SegDataset_Test, self).__init__()
        self.city_name = []
        _path = []
        for _file in sorted(os.listdir(dir)):
            _path += sorted(glob.glob(os.path.join(dir, _file, "*.tif")))
            self.city_name.extend([_file]*len(glob.glob(os.path.join(dir, _file, "*.tif"))))

        self.paths = _path
        self.transform = transform
        
        print(f"Totally {len(self.paths)} patches for testing")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        H, W, _ = np.array(img).shape
        H, W = (H//64)*64, (W//64)*64
        Crop = transforms.CenterCrop((H, W))
        crop_img = Crop(img)
        if self.transform:
            crop_img = self.transform(crop_img)
        basename = os.path.basename(self.paths[item])
        return (crop_img, self.city_name[item], basename)

class Dataset_aggregation(Dataset):
    def __init__(self, dir_cassini, dir_gt_forest, dir_gt_hydro, dir_gt_road, dir_gt_town, transform = None):
        super(Dataset_aggregation, self).__init__()
        dirs = [dir_cassini, dir_gt_forest, dir_gt_hydro, dir_gt_road, dir_gt_town]
        _paths = []
        self.city_name = []
        for _dir in dirs:
            _path = []
            for _file in sorted(os.listdir(_dir)):
                _path += sorted(glob.glob(os.path.join(_dir, _file, "*.tif")))
                if _dir == dir_gt_forest:
                    self.city_name.extend([_file]*len(glob.glob(os.path.join(_dir, _file, "*.tif"))))
            _paths.append(_path)

        assert len(_paths[0]) == len(_paths[1])
        assert len(_paths[0]) == len(_paths[2])
        assert len(_paths[0]) == len(_paths[3])
        assert len(_paths[0]) == len(_paths[4])
        assert len(_paths[0]) == len(self.city_name)
        self.paths = list(zip(*_paths))
        self.transform = transform
        
        print(f"Totally {len(self.paths)} tiles")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, item):
        imgs = [Image.open(p) for p in self.paths[item]] 
        H, W, _ = np.array(imgs[0]).shape
        H, W = (H//64)*64, (W//64)*64
        Crop = transforms.CenterCrop((H, W))
        crop_imgs = [Crop(img) for img in imgs]
        if self.transform:
            crop_imgs = [self.transform(img) for img in crop_imgs]
        basename = os.path.basename(self.paths[item][0])
        return (crop_imgs[0], crop_imgs[1], crop_imgs[2], crop_imgs[3], crop_imgs[4], self.city_name[item], basename)