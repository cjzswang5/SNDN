import os
from torchvision    import transforms
from torch.utils    import data
from .image_utils   import imageLoader, is_image_file, make_dataset

class ImageDataset(data.Dataset):

    def __init__(self, root, transform=None, smalltransform=None,
                 toTensor=None, loader=imageLoader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise Exception('Fond 0 images in ' + root)
        self.root = root
        self.imgs = imgs
        if transform == None:
            raise Exception('transform is None')
        if smalltransform == None:
            raise Exception('smalltransform is None')
        if toTensor == None:
            raise Exception('toTensor is None')
        self.transform = transform
        self.smalltransform = smalltransform
        self.toTensor = toTensor
        self.loader = loader

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = self.loader(path, mode='YCbCr')
        img = self.transform(img)
        img_small = self.smalltransform(img)
        img = self.toTensor(img)
        img_small = self.toTensor(img_small)
        return img, img_small

    def __len__(self):
        return len(self.imgs)

class testImageDataset(data.Dataset):

    def __init__(self, root):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise Exception('Fond 0 images in ' + root)
        self.root = root
        self.imgs = imgs
        self.transform = transforms.ToTensor()
        self.loader = imageLoader

    def __getitem__(self, idx):
        path = self.imgs[idx]
        _, img_name = os.path.split(path)
        img_name, _ = os.path.splitext(img_name)
        img = self.loader(path, mode='RGB')
        img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imgs)
