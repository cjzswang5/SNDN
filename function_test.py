# It's a script to test functions I worted,
# so it is not related to the main program.

from utils.utils import *
import os

import torch

def test_make_dataset():
    image_path = '/home/huzhuoliang/dataset/RESIDE-beta/OTS_BETA/clear'
    file_path = os.path.join(image_path, '2201.jpg')

    print(image_path)
    print(file_path)

    images = pix2pix.make_dataset(image_path)
#    images = pix2pix.make_dataset(file_path)

    print(images)

def test_default_loader():
    path_dir = '/home/huzhuoliang/dataset/RESIDE-beta/OTS_BETA/clear'
    path_img = os.path.join(path_dir, '2201.jpg')
    image = pix2pix.default_loader(path_img)

    print(type(image))

def test_imageSize():
    '''
    Test number of images big than 256x256 in img_path
    '''
    img_path = '/home/huzhuoliang/dataset/RESIDE-beta/OTS_BETA/clear'
    number_big_than_256 = 0
    number_small_than_256 = 0
    for img_name in os.listdir(img_path):
        tmp = imageLoader(os.path.join(img_path, img_name))
        if tmp.size[0] >= 256 and tmp.size[1] > 256:
            number_big_than_256 += 1
        else:
            number_small_than_256 += 1

    print(number_big_than_256)
    print(number_small_than_256)

def test_getPSNR():
    images_raw = torch.rand(6, 3, 8, 8)
    images_noise = torch.rand(6, 3, 8, 8)

    psnr = getPSNR(images_raw, images_noise)
    print(psnr)
    print(psnr.size())

#test_getPSNR()

def test_blurImages():
    image = torch.randn(6, 3, 8, 8)
    blurimg = image
    for i in range(1000):
        blurimg = blurImages(blurimg, kernel_size=3)
    print(blurimg[0, 0])
    print(blurimg.size())

#test_blurImages()

def test_getPSNR_blurImage():
    image = torch.ones(6, 3, 8, 8).float()
    blurimg = blurImages(image)
    print(getPSNR(image, blurimg))

#test_getPSNR_blurImage()

def test_RESIDE_BETA_OTS():
    from utils.resideDataset import RESIDE_BETA_OTS as dataset
    from torchvision import transforms

    root = '/home/huzhuoliang/dataset/RESIDE-beta/OTS_BETA'
    transform_list = []
    transform_list.append(transforms.Resize(256))
    transform_list.append(transforms.CenterCrop(256))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    a = dataset(root, train=False, transform=transform)
    print(len(a))

#    for i, (haze, clear, depth, atmo) in enumerate(a):
#        print(i, haze.size(), clear.size(), depth.size(), atmo.size())

#test_RESIDE_BETA_OTS()

def test_resideDataLoader():
    root = '/home/huzhuoliang/dataset/RESIDE-beta/OTS_BETA'
    dataloader = resideDataLoader(root, 256, 229, workers=4)

    print('dataloader length is {}'.format(len(dataloader)))
    for i, data in enumerate(dataloader):
        haze, clear, depth, atmo = data
        print(i)
        print(haze.size())
        print(clear.size())
        print(depth.size())
        print(atmo.size())

#test_resideDataLoader()

def t_psnr_float(root1, root2):
    from PIL import Image
    from torchvision import transforms
    img1 = Image.open(root1)
    img2 = Image.open(root2)

    t1 = transforms.ToTensor()(img1)
    t2 = transforms.ToTensor()(img2)

    tmp = (t1 - t2) ** 2
    mse = tmp.sum() / (tmp.size(0) * tmp.size(1) * tmp.size(2))
    mse = 1.0 / mse
    return 10 * mse.log10()

def t_psnr_int(root1, root2):
    from PIL import Image
    from torchvision import transforms
    img1 = Image.open(root1)
    img2 = Image.open(root2)

    t1 = transforms.ToTensor()(img1) * 255.0
    t2 = transforms.ToTensor()(img2) * 255.0

    tmp = (t1 - t2) ** 2
    mse = tmp.sum() / (tmp.size(0) * tmp.size(1) * tmp.size(2))
    mse = (255.0 ** 2) / mse
    return 10 * mse.log10()

def t_ssim_float(root1, root2):
    from PIL import Image
    from torchvision import transforms
    img1 = Image.open(root1).convert('L')
    img2 = Image.open(root2).convert('L')

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

def test_PSNR():
    root1 = '/home/huzhuoliang/dataset/RESIDE-beta/OTS_BETA/clear/3189.jpg'
    root2 = '/home/huzhuoliang/dataset/RESIDE-beta/OTS_BETA/haze/3189_1_0.04.jpg'

    from utils.utils import countSSIM
    print(countSSIM(root1, root2))
#    print(t_psnr_float(root1, root2))
#    print(t_psnr_int(root1, root2))
#    print(t_ssim_float(root1, root2))
#    print(t_ssim_float(root2, root1))

#test_PSNR()

def test_resideh5Dataset():
    from utils.resideh5Dataset import resideDataset

    root = 'train_data.h5'
    dataset = resideDataset(root)

    for i, data in enumerate(dataset):
        print(i)
        haze, clear, depth, atmo = data
        #print(haze.size())
        #print(clear.size())
        #print(depth.size())
        #print(atmo.size())

#test_resideh5Dataset()

#def test_syntheticImage():
#    from utils.image_utils import imageLoader, saveTensorToImage
#
#    A = 1.0
#    beta = 0.4
#
#

def test_resideDataset_ITS():
    from utils.resideDataset import RESIDE_STANDARD_ITS

    root = '/home/huzhuoliang/dataset/RESIDE/RESIDE-Standard/ITS'
    dataset = RESIDE_STANDARD_ITS(root, train=False)

    for tmp in enumerate(dataset):
        print(tmp)

#test_resideDataset_ITS()
