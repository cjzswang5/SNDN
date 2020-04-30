import os, re, random, argparse, torch
import numpy as np
from torch       import nn, optim
from models.G    import Generator
from utils.utils import resideh5DataLoader, str_green_time
from tqdm        import tqdm

parser = argparse.ArgumentParser()

# train data file. must be a .h5 file. Can be created use create_train.py.
parser.add_argument('--root', type=str, default='')
# number of train epoch.
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--batchSize', type=int, default=35)

parser.add_argument('--cuda', action='store_true')
# path to save model files.
parser.add_argument('--savePath', type=str, default='/output')
# path to load model files.
parser.add_argument('--net', type=str, default='')
# optimizer learning rate.
parser.add_argument('--lr', type=float, default=0.0005)

parser.add_argument('--seed', type=int, default=1)

opt = parser.parse_args()

# set seed for train
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)    # if you are using multi-GPU.
np.random.seed(opt.seed)                    # Numpy module.
random.seed(opt.seed)                       # Python random module.
torch.manual_seed(opt.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

dataLoader = resideh5DataLoader(opt.root, batchSize=opt.batchSize)

netG = Generator()

criterionMSE = nn.MSELoss()

if opt.cuda:
    criterionMSE.cuda()
    netG.cuda()

optimizer = optim.Adam(netG.parameters(), lr=opt.lr)

base_epoch = 0
if opt.net != '':
    state = torch.load(opt.net)
    netG.load_state_dict(state['netG'])
    optimizer.load_state_dict(state['optim'])
    base_epoch = state['epoch']

netG.train()

print(str_green_time() + ' start training!')
end_epoch = base_epoch + opt.epochs
tqdm_bar1 = tqdm(range(base_epoch, end_epoch), ncols=80)
tqdm_bar1.set_description('epoch[  1] avg_loss[?.????????]')
for epoch in tqdm_bar1:

    loss_list = []
    tqdm_bar2 = tqdm(dataLoader, ncols=80, leave=False)
    tqdm_bar2.set_description('loss[?.????????]')
    for i, data in enumerate(tqdm_bar2, 0):
        optimizer.zero_grad()

        netG.zero_grad()

        haze, clear = data
        if opt.cuda:
            haze  =  haze.cuda()
            clear = clear.cuda()

        output, _ = netG(haze)

        loss = criterionMSE(clear, output)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        tqdm_bar2.set_description('loss[{:.8f}]'.format(loss.item()))

    desc_str = 'epoch[{:3d}] avg_loss[{:.8f}]'
    desc_str = desc_str.format(epoch+1, np.mean(loss_list))
    tqdm_bar1.set_description(desc_str)


#        if (i+1) % 50 == 0 or (i+1) == len(dataLoader):
#            print(str_green_time() + \
#                  ' eopch[{:3d}/{:3d}]'.format(epoch+1, end_epoch) + \
#                  ' batch[{:3d}/{:3d}]'.format(i+1, len(dataLoader)) + \
#                  ' loss[{:.8f}]'.format(loss.item()))
    if (epoch+1) % 10 == 0:
#        print(str_green_time(True) + \
#              ' epoch[{:3d}]'.format(epoch+1) + \
#              ' avg_loss{:.4f}'.format(np.mean(loss_list)))
        save_name = 'OITS_color_{:03d}.pth'.format(epoch+1)
        save_path = os.path.join(opt.savePath, save_name)
        state = {'netG': netG.state_dict(),
                 'optim': optimizer.state_dict(),
                 'epoch': epoch+1}
        torch.save(state, save_path)

print('train done!')

