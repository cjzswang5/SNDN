import torch
import torch.utils.data as data
from .image_utils import loadH5

class resideh5Dataset(data.Dataset):
    '''
    Class to load trainning set.
      * root is the path to h5 data file.
    '''
    def __init__(self, root):

        self.root = root

    def __getitem__(self, idx):
        with loadH5(self.root) as f:
            haze_image  = torch.tensor(f['haze' ][idx, ...])
            clear_image = torch.tensor(f['clear'][idx, ...])
        return haze_image, clear_image

    def __len__(self):
        with loadH5(self.root) as f:
            return f['clear'].shape[0]

