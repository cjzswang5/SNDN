import os, cv2, h5py, re
import numpy as np
from tqdm import tqdm

def list_match(l, p):
    return [i for i in l if re.match(p, i) is not None]

def readDepth(path_depth):
    _, ext = os.path.splitext(path_depth)
    r_depth = None
    if ext == '.png':
        r_depth = cv2.imread(path_depth, 0).astype('float32') / 255.0
    elif ext == '.mat':
        r_depth = h5py.File(path_depth, 'r')['depth'][:].astype('float32')
        r_depth = np.swapaxes(r_depth, 0, 1)
        r_depth = (0.0 - r_depth) * 0.2
        r_depth = np.exp(r_depth)
    else:
        raise Exception('depth path uncorrect!')
    return r_depth

def hazeSingleImage(image, depth, A):
    '''
    haze single image.
        image [ndarray] -  clear image, shape is 3xWxH, range is [0, 1].
        depth [ndarray] -  depth image, shape is WxH, range is [0, 1].
    '''
    t = depth[np.newaxis, ...].repeat(3, axis=0)
    p1 = image * t
    t = 1 - t
    t[0] *= A[2]
    t[1] *= A[1]
    t[2] *= A[0]
    return p1 + t

def genHazeImage(path_in, path_depth, path_out, A):
    img_clear = cv2.imread(path_in).astype('float32') / 255.0
    img_clear = img_clear.transpose(2, 0, 1)
    img_depth = readDepth(path_depth)
    r = hazeSingleImage(img_clear, img_depth, A)
    r = (r * 255).astype('int').transpose(1, 2, 0)
    cv2.imwrite(path_out, r)


def getAirlights():
    tmp  = []
    with open('result.txt', 'r') as f:
        tmp = f.read().split('\n')[:-1]
    r = []
    for t in tmp:
        [_, tr, tg, tb] = t.split(' ')
        r.append([float(tr), float(tg), float(tb)])
    return r

def genHazeImageDir(root_in, root_depth, root_out):
    airlights = getAirlights()
    l_depth = os.listdir(root_depth)
    l_clear = os.listdir(root_in)
    for tmp in tqdm(l_clear, leave=False, ncols=80):
        clear_name = tmp
        imgName, _ = os.path.splitext(tmp)
        depth_name = list_match(l_depth, '^'+imgName+'[_.]')[0]
        path_clear = os.path.join(root_in,    clear_name)
        path_depth = os.path.join(root_depth, depth_name)
        for airlight in airlights:
            haze_name = '{}_{:.2f}_{:.2f}_{:.2f}.png'.format(imgName, *airlight)
            path_out = os.path.join(root_out, haze_name)
            genHazeImage(path_clear, path_depth, path_out, airlight)

root_ITS = '/data/huzhuoliang/RESIDE/RESIDE_unzip/ITS_v2'
root_ITS_clear = os.path.join(root_ITS, 'clear')
root_ITS_depth = os.path.join(root_ITS, 'trans')
root_ITS_out   = os.path.join(root_ITS, 'hazy_color')

root_OTS = '/data/huzhuoliang/RESIDE/RESIDE_unzip/OTS_BETA'
root_OTS_clear = os.path.join(root_OTS, 'clear')
root_OTS_depth = os.path.join(root_OTS, 'depth')
root_OTS_out   = os.path.join(root_OTS, 'haze_color')

#i = 0
#for airlight in getAirlights():
#    i += 1
#    path_out = 'haze_results/{:d}_haze_{:.1f}_{:.1f}_{:.1f}.png' \
#        .format(i, *airlight)
#    r = genHazeImage(root_clear, root_depth2, path_out, airlight)
#    print('{:.<70s}done!'.format(path_out))

genHazeImageDir(root_ITS_clear, root_ITS_depth, root_ITS_out)
genHazeImageDir(root_OTS_clear, root_OTS_depth, root_OTS_out)

