import os, h5py, torch
from utils.utils import resideDataLoader, str_green_time
from tqdm import tqdm

def create_train_h5(args):
    file_name, root, imageSize = args

    dataloader = resideDataLoader(root, imageSize)

    dl = len(dataloader)
    print('dataset have {} elements.'.format(dl))

    h5_file = h5py.File(file_name, 'w-')

    # haze and clear images have 3 channels,
    tmp = (dl, 3, imageSize, imageSize)
    h5_file.create_dataset('haze',  tmp, dtype='float32')
    h5_file.create_dataset('clear', tmp, dtype='float32')

    title = '{:18s}'.format(file_name)
    for i, data in enumerate(tqdm(dataloader, ncols=80, desc=title)):
        haze, clear = data
        # I searched 'how to create h5 file', and it tells me the three dots
        # (...) can not be ommited.
        h5_file[ 'haze'][i, ...] = haze
        h5_file['clear'][i, ...] = clear

    h5_file.close()
    print('{} done'.format(file_name))

imageSize = 256
h5_root = '/data/huzhuoliang/RESIDE/h5_color/'
root = ((h5_root + 'ITS_train.h5', 'txt/ITS/train.txt', imageSize),
        (h5_root + 'ITS_test.h5',  'txt/ITS/test.txt',  imageSize),
        (h5_root + 'OTS_train.h5', 'txt/OTS/train.txt', imageSize),
        (h5_root + 'OTS_test.h5',  'txt/OTS/test.txt',  imageSize),
        (h5_root + 'OITS_train.h5','txt/OITS/train.txt',imageSize),
        (h5_root + 'OITS_test.h5', 'txt/OITS/test.txt', imageSize))
# I need create train and test.
for tmp in root:
    create_train_h5(tmp)

