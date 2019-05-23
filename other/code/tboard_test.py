from tensorboardX import SummaryWriter

import torch
import torch.nn as nn


name = 'test_2'

def tester():
    log = SummaryWriter('runs/{}'.format(name))
    
    for i in range(500):
        log.add_scalar('test', i, i)

    for i in range(600):
        log.add_scalar('another_test', i, i)
    



if __name__ == '__main__':
    tester()