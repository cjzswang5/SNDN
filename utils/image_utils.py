import os
from h5py                   import File
from PIL                    import Image
from torch                  import tensor
from torchvision.transforms import ToPILImage

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png',
                  '.PNG', '.ppm', '.PPM',  '.bmp',  '.BMP']

def loadH5(path):
    return File(path, 'r')

def loadTrans(path):
    f = File(path, 'r')
    return tensor(f['depth']).T.float()

def loadTransPIL(path):
    tsr = loadTrans(path)
    tsr_int = (tsr - tsr.min()) / tsr.max()
    return ToPILImage()(tsr_int)

def imageLoader(img_path, mode='RGB'):
    return Image.open(img_path).convert(mode)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(root):
    images = []
    for image_name in os.listdir(root):
        if is_image_file(image_name):
            full_path = os.path.join(root, image_name)
            images.append(full_path)
    return images

def saveTensorToImage(tsr, path, mode='RGB'):
    img = ToPILImage(mode)(tsr)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(path)

def saveTransToImage(tsr, path):
    tsr_int = (((tsr - tsr.min()) / tsr.max()) * 255).int()
    saveTensorToImage(tsr_int, path, mode='I')

def saveBatchTensorToImage(tsr, path, number, name='result', mode='RGB'):
    i = 0
    for tmp in tsr:
        save_path = os.path.join(path, '{}_{:05d}.png'.format(name, number+i))
        saveTensorToImage(tmp, save_path, mode=mode)
        i += 1

def saveBatchTransToImage(tsr, path, number, name='Trans'):
    i = 0
    for tmp in tsr:
        save_path = os.path.join(path, '{}_{:05d}.png'.format(name, number+i))
        saveTransToImage(tmp, save_path)
        i += 1

