import os, csv
import numpy as np
from PIL                import Image
from tqdm               import tqdm
from prettytable        import from_csv
from skimage.measure    import compare_psnr, compare_ssim

def evaluation(root1, root2, title):
    '''
    root1 must be ground truth images directory.
    root2 must be hazy images directory.
    '''
    l_psnr = []
    l_ssim = []
    i = 0
    test_list = os.listdir(root2)
    if len(test_list) == 0:
        raise Exception('dehaze dir is empty.')
    for tmp2 in tqdm(test_list, leave=False, ncols=80, desc=title):
        i += 1
        #print('[{:4d}/{:4d}] {}'.format(i, len(test_list), tmp2))
        name, _ = os.path.splitext(tmp2)
        img_path1 = os.path.join(root1, name[:4]+'.png')
        img_path2 = os.path.join(root2, tmp2)
        psnr, ssim = eval2img(img_path1, img_path2)
        l_psnr.append(psnr)
        l_ssim.append(ssim)
    #print('PSNR is {:.2f}'.format(np.mean(l_psnr)))
    #print('SSIM is {:.4f}'.format(np.mean(l_ssim)))
    return np.mean(l_psnr), np.mean(l_ssim)

def eval2img(img_path1, img_path2):
    '''
    Given two images path, return PSNR and SSIM between this two images.
    '''
    img1 = np.array(Image.open(img_path1))
    img2 = np.array(Image.open(img_path2))
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(img1, img2, multichannel=True)
    return psnr, ssim

SOTS_root = '/home/huzhuoliang/dataset/RESIDE/RESIDE-Standard/SOTS'
root_indoor = 'indoor'
root_outdoor = 'outdoor'
root_gt = 'gt'
root_dehaze = 'dehaze'

# evaluation indoor test set.
root1 = os.path.join(SOTS_root, root_indoor, root_gt)
root2 = os.path.join(SOTS_root, root_indoor, root_dehaze)
psnr1, ssim1 = evaluation(root1, root2, '[1/2] indoor ')

# evaluation outdoor test set.
root1 = os.path.join(SOTS_root, root_outdoor, root_gt)
root2 = os.path.join(SOTS_root, root_outdoor, root_dehaze)
psnr2, ssim2 = evaluation(root1, root2, '[2/2] outdoor')

tb1 = ['', 'PSNR', 'SSIM']
tb2 = ['indoor', '{:.2f}'.format(psnr1), '{:.4f}'.format(ssim1)]
tb3 = ['outdoor', '{:.2f}'.format(psnr2), '{:.4f}'.format(ssim2)]

# write to csv file.
csv_name = 'e_result.csv'
with open(csv_name, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(tb1)
    writer.writerow(tb2)
    writer.writerow(tb3)

# read from csv file and print.
with open(csv_name, 'r') as f:
    tb = from_csv(f)
    print(tb)

print('Data writed into {}.'.format(csv_name))

