import torch
import os
print('{:^10s} | {:^9s} | {:^6s} | {:^9s}'.format('name', 'param', 'acc', 'epoch'))
print('-------------------------------------------')
for root, dirs, files in os.walk("./checkpoint", topdown=False):
    for name in files:
        surffix = os.path.splitext(name)[-1]
        if surffix == '.pth':
            ckpt = torch.load(os.path.join(root, name))
            print('{:>10s} | {:>9d} | {:0<2.2f}% | epoch {}'.format(ckpt['name'], ckpt['total_param'], ckpt['acc'], ckpt['epoch']))