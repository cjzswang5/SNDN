import scipy.io as scio
import torch, time

from torch.utils.data import DataLoader
from torchvision      import transforms
from .imageDataset    import ImageDataset, testImageDataset
from .resideDataset   import RESIDE_TXT
from .resideh5Dataset import resideh5Dataset
from .image_utils     import imageLoader

def getTimeStamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def str_green(s, highlight=False):
    c = 1 if highlight else 0
    return '\033[{:d};32m'.format(c) + s + '\033[0m'

def str_green_time(highlight=False):
    return str_green(getTimeStamp(), highlight)

def imageDataLoader(root, originalSize, imageSize, small_imageSize,
                    batchSize=32, workers=4, shuffle=True):
    dataset = ImageDataset(root=root,
                           transform=transforms.Compose([
                               transforms.Resize(originalSize),
                               transforms.RandomCrop(imageSize)]),
                           smalltransform=transforms.Compose([
                               transforms.Resize(small_imageSize),
                               transforms.Resize(imageSize, interpolation=0)]),
                           toTensor=transforms.Compose([
                               transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batchSize,
                            shuffle=shuffle, num_workers=int(workers))
    return dataloader

def testImageDataLoader(root, workers=4):
    dataset = testImageDataset(root=root)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=int(workers))
    return dataloader

def resideDataLoader(root, imageSize, batchSize=1, workers=4, shuffle=True):
    transform_list = []
    transform_list.append(transforms.Resize(imageSize))
    transform_list.append(transforms.CenterCrop(imageSize))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    dataset = RESIDE_TXT(root, transform)

    dataloader = DataLoader(dataset, batch_size=batchSize,
                            shuffle=shuffle, num_workers=int(workers))
    return dataloader

def resideh5DataLoader(root, batchSize=32, workers=8, shuffle=True):
    dataset = resideh5Dataset(root)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle,
                            num_workers=int(workers))
    return dataloader

def getPSNR(images_raw, images_noise):
    '''
    Count PSNR for a batch of images.
      * Input images must be 4 dimension tensor(BxCxHxW).
      * Input images must in the range of [0, 1].
      * Output is a float PSNR array(Bx1).
      * image_raw and images_noise must be same size.
    '''
    diff = images_raw - images_noise
    mse = torch.mul(diff, diff).mean((1, 2, 3))
    psnr = 10.0 * torch.log(1.0 / mse).unsqueeze(1)
    return psnr

def blurImages(image, kernel_size=3):
    '''
    Blur image use mean filter.
      * Input images must be 4 dimension tensor(BxCxHxW).
      * kernel_size nust be odd number.
    '''
    if kernel_size % 2 == 0:
        raise Exception('kernel_size must be odd number')
    paddingSame = torch.nn.ReplicationPad2d(int((kernel_size - 1) / 2))
    conv2d = torch.nn.Conv2d(3, 3, kernel_size, 1, 0, bias=False)
    conv2d.weight.requires_grad = False
    conv2d.weight.data = torch.ones((3, 3, kernel_size, kernel_size)).float()
    blured_image = conv2d(paddingSame(image)) / (3 * (kernel_size ** 2 ))
    return blured_image

def countSSIM(root1, root2):
    '''
    Count SSIM between root1 and root2.
      * root1 and root2 is the full path of test image.
    '''
    img1 = imageLoader(root1, 'L')
    img2 = imageLoader(root2, 'L')

    t1 = transforms.ToTensor()(img1).float() / 255.0
    t2 = transforms.ToTensor()(img2).float() / 255.0

    u1 = (t1.sum() / (t1.size(1) * t1.size(2))).float()
    u2 = (t2.sum() / (t2.size(1) * t2.size(2))).float()

    u1_m = torch.ones(t1.size()).float() * u1
    u2_m = torch.ones(t2.size()).float() * u2

    sigm1 = ((t1 - u1_m) ** 2).sum() / (t1.size(1) * (t1.size(2) - 1))
    sigm2 = ((t2 - u2_m) ** 2).sum() / (t2.size(1) * (t2.size(2) - 1))

    tmp = (t1 - u1_m) * (t2 - u2_m)
    sigm12 = tmp.sum() / (tmp.size(1) * (tmp.size(2) - 1))

    L = 1.0
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    ssim = (2*u1*u2 + c1) * (2*sigm12 + c2)
    ssim /= (u1**2 + u2**2 + c1) * (sigm1 + sigm2 + c2)

    return ssim.item()
