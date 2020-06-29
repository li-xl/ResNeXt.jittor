#coding=utf-8

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

'''
convert from https://github.com/prlz77/ResNeXt.pytorch
'''

import argparse
import json
from PIL import Image
import os
import numpy as np
import jittor as jt
import jittor.transform as transform
from resnext import CifarResNeXt
from cifar import CIFAR10,CIFAR100
import time
import random

class ToTensor:
    def __call__(self,img):
        if isinstance(img, Image.Image):
            return np.array(img).transpose((2,0,1)) / np.float32(255)
        return img

def constant_pad(img,padding,value):
    if isinstance(padding, int):
        pl = padding
        pr = padding
        pt = padding
        pb = padding
    else:
        pl, pr, pt, pb = padding

    w,h = img.size
    p = Image.new(img.mode, (w+pl+pr, h+pt+pb), (0, 0, 0))
    #print(p.size,img.size,img.mode,p.mode)
    p.paste(img, (pl, pt, pl+w, pt+h))
    return p

def crop(img, top, left, height, width):
    '''
    Function for cropping image.

    Args::

        [in] img(Image.Image): Input image.
        [in] top(int): the top boundary of the cropping box.
        [in] left(int): the left boundary of the cropping box.
        [in] height(int): height of the cropping box.
        [in] width(int): width of the cropping box.

    Example::
        
        img = Image.open(...)
        img_ = transform.crop(img, 10, 10, 100, 100)
    '''
    return img.crop((left, top, left + width, top + height))

class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
 
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
 
    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w,h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
 
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
 
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = constant_pad(img, self.padding, self.fill)
 
        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = constant_pad(img, (self.size[1] - img.size[0], 0), self.fill)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = constant_pad(img, (0, self.size[0] - img.size[1]), self.fill)
 
        i, j, h, w = self.get_params(img, self.size)
 
        return crop(img, i, j, h, w)


# train function (forward, backward, update)
def train(net,optimizer,train_data,state):
    net.train()
    loss_avg = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = jt.array(data), jt.array(target)

        # forward
        output = net(data)

        loss = jt.nn.cross_entropy_loss(output, target)
        optimizer.step(loss)
        # exponential moving average
        loss_avg = loss_avg * 0.2 + float(loss.data[0]) * 0.8
    end_time = time.time()
    fps = (len(train_data)*train_data.batch_size)/(end_time-start_time)

    state['train_loss'] = loss_avg
    state['train_fps'] = fps


# test function (forward only)
def test(net,test_data,state):
    net.eval()
    loss_avg = 0.0
    correct = 0
    start_time=time.time()
    for batch_idx, (data, target) in enumerate(test_data):
        data, target = jt.array(data), jt.array(target)

        # forward
        output = net(data)
        loss = jt.nn.cross_entropy_loss(output, target)
        
        # accuracy
        pred = jt.argmax(output,dim=1)[0]
        correct += float(jt.sum(pred==target).data[0])


        # test loss average
        loss_avg += float(loss.data[0])
    end_time = time.time()
    fps = (len(test_data)*test_data.batch_size)/(end_time-start_time)

    state['test_loss'] = loss_avg / len(test_data)
    state['test_accuracy'] = correct / (len(test_data)*test_data.batch_size)
    state['test_fps'] = fps


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=128)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='~/output_logs/ResNeXt.jittor/models', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='~/output_logs/ResNeXt.jittor/logs/', help='Log folder.')
    args = parser.parse_args()
    
    jt.flags.use_cuda=1
    # Init logger
    args.log = os.path.expanduser(args.log)
    args.save = os.path.expanduser(args.save)
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    args.epochs = args.epochs * 128 // args.batch_size
    args.schedule = [x * 128 // args.batch_size for x in args.schedule]

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transform.Compose(
        [transform.RandomHorizontalFlip(), RandomCrop(32, padding=4), ToTensor(),
         transform.ImageNormalize(mean, std)])
    test_transform = transform.Compose(
        [ToTensor(), transform.ImageNormalize(mean, std)])

    
    if args.dataset == 'cifar10':
        train_data = CIFAR10(args.data_path, train=True, transform=train_transform,batch_size=args.batch_size,num_workers=args.prefetch)
        test_data = CIFAR10(args.data_path, train=False, transform=test_transform,shuffle=False,num_workers=args.prefetch,batch_size=args.test_bs)
        nlabels = 10
    else:
        train_data = CIFAR100(args.data_path, train=True, transform=train_transform, download=True,batch_size=args.batch_size,num_workers=args.prefetch)
        test_data = CIFAR100(args.data_path, train=False, transform=test_transform, download=True,shuffle=False,num_workers=args.prefetch,batch_size=args.test_bs)
        nlabels = 100

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)

    optimizer = jt.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    # Main loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        if epoch in args.schedule:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

        state['epoch'] = epoch

        train(net,optimizer,train_data,state)
        test(net,test_data,state)

        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            net.save(os.path.join(args.save, 'CifarResNeXt.jittor'))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print("train_loss:",state['train_loss'])
        print("test_loss:",state['test_loss'])
        print("test_accuracy:",state["test_accuracy"])
        print("train_fps:",state["train_fps"])
        print("test_fps:",state["test_fps"])
        print("Best accuracy: %f" % best_accuracy)

    log.close()
