import os
import torch
import random
from itertools import product
import torch.utils.data as data
from .image_utils import imageLoader, loadTransPIL

# This class is used for RESIDE-beta OTS.
class RESIDE_BETA_OTS(data.Dataset):
    '''
    Class to load RESIDE-beta trainning set.
      * root must have three directory: clear, depth and haze.
    '''
    def __init__(self, root, train=True, transform=None, seed=None):
        ATMO = [0.8, 0.85, 0.9, 0.95, 1]
        BETA = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2]
        if seed is not None:
            random.seed(seed)

        self.root = root
        self.clear_root = os.path.join(self.root, 'clear')
        self.depth_root = os.path.join(self.root, 'depth')
        self.haze_root  = os.path.join(self.root,  'haze')
        self.transform = transform
        self.haze = []
        self.images = []

        # train set and test set are splited in 2 txt files.
        file_name = 'train.txt'
        if not train:
            file_name = 'test.txt'
            BETA = [0.2]
        f = open(os.path.join(root, file_name), 'r')
        tmp_names = f.read().splitlines()
        f.close()
        for tmp in tmp_names:
            tmp_name, tmp_ext = os.path.splitext(tmp)
            self.images.append(tmp_name)
        for tmp in product(self.images, ATMO, BETA):
            tmp_haze = '{}_{}_{}.jpg'.format(*tmp)
            self.haze.append((tmp_haze, tmp[0], tmp[1]))
        random.shuffle(self.haze)

    def __getitem__(self, idx):
        haze_n, clear_n, atmo = self.haze[idx]
        # depth image must be .mat file, clear and haze must be .jpg file.
        haze_name  = os.path.join(self.haze_root,  haze_n)
        clear_name = os.path.join(self.clear_root, clear_n+'.jpg')

        haze_image  = imageLoader(haze_name)
        clear_image = imageLoader(clear_name)
        if self.transform is not None:
            haze_image  = self.transform(haze_image)
            clear_image = self.transform(clear_image)

        return haze_image, clear_image

    def __len__(self):
        # haze, clear, depth have same length.
        return len(self.haze)

# This class is used for RESIDE-Standard ITS.
# and ... ... without depth/trans.
class RESIDE_STANDARD_ITS(data.Dataset):
    '''
    Class to load RESIDE-Strandard indoor trainning set.
      * root must have three directory: clear, trans and hazy.
    '''
    def __init__(self, root, train=True, transform=None, seed=None):
        if seed is not None:
            random.seed(seed)

        self.root = root
        self.clear_root = os.path.join(self.root, 'clear')
        self.haze_root  = os.path.join(self.root,  'hazy')
        self.transform = transform
        self.haze = []
        self.images = []

        # train set and test set are splited in 2 txt files.
        file_name = 'train.txt'
        if not train:
            file_name = 'test.txt'
        self.haze = self.find_list_by_txt_file(os.path.join(root, file_name))

    def find_list_by_txt_file(self, path):
        r = []
        with open(path, 'r') as f:
            t = f.readlines()
            for tmp in t:
                t1, t2 = tmp.split(' ')
                r.append((t1, t2[:-1]))
        return r

    def __getitem__(self, idx):
        clear_n, haze_n = self.haze[idx]
        # depth image must be .mat file, clear and haze must be .jpg file.
        haze_name  = os.path.join(self.haze_root,  haze_n)
        clear_name = os.path.join(self.clear_root, clear_n)

        haze_image  = imageLoader(haze_name)
        clear_image = imageLoader(clear_name)
        if self.transform is not None:
            haze_image  = self.transform(haze_image)
            clear_image = self.transform(clear_image)
        return haze_image, clear_image

    def __len__(self):
        # haze, clear, depth have same length.
        return len(self.haze)

class RESIDE_TXT(data.Dataset):
    '''
    Class to load RESIDE data set.
      * root is the txt file for data root.
    '''
    def __init__(self, root, transform=None):
        self.l_clear = []
        self.l_haze  = []
        self.transform=transform
        for i in open(root, 'r'):
            tmp = i[:-1].split(' ', 1)
            self.l_clear.append(tmp[0])
            self.l_haze.append(tmp[1])

    def __getitem__(self, idx):
        clear_name = self.l_clear[idx]
        haze_name  = self.l_haze[idx]

        haze_image  = imageLoader(haze_name)
        clear_image = imageLoader(clear_name)
        if self.transform is not None:
            haze_image  = self.transform(haze_image)
            clear_image = self.transform(clear_image)
        return haze_image, clear_image

    def __len__(self):
        return len(self.l_clear)

