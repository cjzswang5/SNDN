import os, random, re
from itertools import product

def list_match(l, p):
    return [i for i in l if re.match(p, i) is not None]

def select_colorhaze(l_clear, clear_path, haze_path):
    l_haze  = os.listdir(haze_path)
    r = []
    for imgName in l_clear:
        imgN, _ = os.path.splitext(imgName)
        imgH = list_match(l_haze, '^'+imgN+'_')
        for tmp_imgH in imgH:
            cp = os.path.join(clear_path, imgName)
            hp = os.path.join(haze_path,  tmp_imgH)
            r.append((cp, hp))
    return r

def comb_OTS(l_clear, clear_path, haze_path, ATMO, BETA):
    l_haze, images, l_r = [], [], []
    for tmp in l_clear:
        name, _ = os.path.splitext(tmp)
        images.append(name)
    for tmp in product(images, ATMO, BETA):
        l_haze.append((tmp[0] + '.jpg', '{}_{}_{}.jpg'.format(*tmp)))
    random.shuffle(l_haze)
    for tmp in l_haze:
        p1 = os.path.join(clear_path, tmp[0])
        p2 = os.path.join(haze_path, tmp[1])
        l_r.append((p1, p2))
    return l_r

def list_OTS(root, num_train, num_test):
    clear_path     = os.path.join(root, 'clear')
    haze_path      = os.path.join(root, 'haze')
    hazecolor_path = os.path.join(root, 'haze_color')
    l_clear = os.listdir(clear_path)
    if num_train+num_test > len(l_clear):
        raise Exception('Insufficient pictures.')
    l_clear = random.sample(l_clear, num_train+num_test)
    l_clear_train = l_clear[:num_train]
    l_clear_test  = l_clear[num_train:]
    ATMO = [0.8, 0.85, 0.9, 0.95, 1]
    BETA = [0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2]
    lr_train  = comb_OTS(l_clear_train, clear_path, haze_path, ATMO, BETA)
    lr_train += select_colorhaze(l_clear_train, clear_path, hazecolor_path)
    ATMO, BETA = [1], [0.2]
    lr_test   = comb_OTS(l_clear_test, clear_path, haze_path,  ATMO, BETA)
    lr_test  += select_colorhaze(l_clear_test,  clear_path, hazecolor_path)
    random.shuffle(lr_train)
    random.shuffle(lr_test)
    return lr_train, lr_test

def comb_ITS(l_clear, l_haze, clear_path, haze_path, train=True):
    lr = []
    add_str = '_'
    if not train:
        add_str += '10_'
    for tmp in l_clear:
        name, _ = os.path.splitext(tmp)
        lh = list_match(l_haze, '^'+name+add_str)
        for ttmp in lh:
            lr.append((tmp, ttmp))
    random.shuffle(lr)
    r = []
    for tmp in lr:
        p1 = os.path.join(clear_path, tmp[0])
        p2 = os.path.join(haze_path, tmp[1])
        r.append((p1, p2))
    return r

def list_ITS(root, num_train, num_test):
    clear_path = os.path.join(root, 'clear')
    haze_path  = os.path.join(root, 'hazy')
    hazecolor_path  = os.path.join(root, 'hazy_color')
    l_r = []
    l_clear = os.listdir(clear_path)
    l_haze  = os.listdir(haze_path)
    if num_train+num_test > len(l_clear):
        raise Exception('Insufficient pictures.')
    l_clear = random.sample(l_clear, num_train+num_test)
    l_clear_train = l_clear[:num_train]
    l_clear_test  = l_clear[num_train:]
    lr_train  = comb_ITS(l_clear_train, l_haze, clear_path, haze_path, True)
    lr_train += select_colorhaze(l_clear_train, clear_path, hazecolor_path)
    lr_test   = comb_ITS(l_clear_test,  l_haze, clear_path, haze_path, False)
    lr_test  += select_colorhaze(l_clear_test,  clear_path, hazecolor_path)
    random.shuffle(lr_train)
    random.shuffle(lr_test)
    return lr_train, lr_test

def write_txt(l_haze, txt_name):
    with open(txt_name, 'w') as f:
        for tmp in l_haze:
            f.write('{} {}\n'.format(tmp[0], tmp[1]))

def write_txt_foo(root, num_train, num_test, list_foo, txtpath):
    lr_train, lr_test = list_foo(root, num_train, num_test)
    write_txt(lr_train, os.path.join(txtpath, 'train.txt'))
    write_txt(lr_test,  os.path.join(txtpath, 'test.txt' ))
    return len(lr_train), len(lr_test)

def write_txt_both_foo(root_OTS, root_ITS, num_train, num_test, txtpath):
    if num_train % 9 != 0:
        raise Exception('num_train must be multiple of 9.')
    if num_test % 2 != 0:
        raise Exception('num_test must be even number.')
    n1 = int(num_train / 9 * 2)
    n2 = int(num_train / 9 * 7)
    ltr1, lte1 = list_OTS(root_OTS, n1, int(num_test / 2))
    ltr2, lte2 = list_ITS(root_ITS, n2, int(num_test / 2))
    ltr = ltr1 + ltr2
    lte = lte1 + lte2
    random.shuffle(ltr)
    random.shuffle(lte)
    write_txt(ltr, os.path.join(txtpath, 'train.txt'))
    write_txt(lte, os.path.join(txtpath, 'test.txt' ))
    return len(ltr), len(lte)

root_OTS = '/data/huzhuoliang/RESIDE/RESIDE_unzip/OTS_BETA'
root_ITS = '/data/huzhuoliang/RESIDE/RESIDE_unzip/ITS_v2'

random.seed(1)

ltr, lte = write_txt_foo(root_OTS, 340, 20, list_OTS, 'txt/OTS/')
print('OTS  train set has {:5d} images.'.format(ltr))
print('OTS  test  set has {:5d} images.'.format(lte))
ltr, lte = write_txt_foo(root_ITS, 840, 20, list_ITS, 'txt/ITS/')
print('ITS  train set has {:5d} images.'.format(ltr))
print('ITS  test  set has {:5d} images.'.format(lte))
ltr, lte = write_txt_both_foo(root_OTS, root_ITS, 720, 20, 'txt/OITS/')
print('OITS train set has {:5d} images.'.format(ltr))
print('OITS test  set has {:5d} images.'.format(lte))

