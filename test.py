import os, re, random, argparse, torch, time
import numpy as np
from tqdm               import tqdm
from torch              import nn, optim
from models.G           import Generator
from utils.utils        import resideh5DataLoader, str_green_time
from utils.utils        import testImageDataLoader
from utils.image_utils  import saveTensorToImage

parser = argparse.ArgumentParser()

# path to save weight files.
parser.add_argument('net', type=str)
# path for test images.
parser.add_argument('-i', '--inputPath', type=str, default='')
# path to save output images.
parser.add_argument('-o', '--outputPath', type=str, default='')

parser.add_argument('-c', '--cuda', action='store_true')

parser.add_argument('-a', '--concat', action='store_true')


opt = parser.parse_args()

def test_h5(netG, dataLoader, output_dir, use_cuda):
    for i, data in enumerate(dataLoader, 0):
        haze, clear, depth, atmo = data
        if use_cuda:
            haze  =  haze.cuda()
            clear = clear.cuda()
            depth = depth.cuda()
            atmo  =  atmo.cuda()
        output, k = netG(haze)
        concat = torch.cat((haze, output, clear, k), 3).cpu()
        output_path = os.path.join(output_dir, 'r{:03d}.png'.format(i+1))
        progress = '[{:3d}/{:3d}]'.format(i+1, len(dataLoader))
        saveTensorToImage(concat[0], output_path)
        print('{} {} {}'.format(progress, output_path, 'saved !'))

def test_img(netG, dataLoader, output_dir, use_cuda, concat):
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataLoader, ncols=80)):
            img, img_name = data
            if use_cuda:
                img = img.cuda()
            output, k = netG(img)
            if concat:
                output = torch.cat((img, output, k), 3).cpu()
            output_path = os.path.join(output_dir, '{}.png'.format(img_name[0]))
            saveTensorToImage(output[0].cpu(), output_path)

netG = Generator()
state = torch.load(opt.net)
netG.load_state_dict(state['netG'])

if opt.cuda:
    netG.cuda()

netG.eval()
print(str_green_time() + ' start testing!')

# test h5
#dataLoader = resideh5DataLoader(opt.i, batchSize=1)
#test_h5(netG, dataLoader, opt.o, opt.cuda)

# test image
dataLoader = testImageDataLoader(opt.inputPath)
test_img(netG, dataLoader, opt.outputPath, opt.cuda, opt.concat)
print(str_green_time() + ' test done!')

