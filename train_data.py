
# -*- coding: utf-8 -*-
import argparse
import math
from math import log10
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import dataloader
from tensorboardX import SummaryWriter
from sepconv1 import bak2_sepconv as sepconv

import PIL.Image as pimg
import numpy as np
import random


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
        )
        # end
        def Upsample(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        # end
        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
            # end
        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleDeconv4 = Basic(512, 256)
        self.moduleDeconv3 = Basic(256, 128)
        self.moduleDeconv2 = Basic(128, 64)


        self.moduleUpsample5 = Upsample(512, 512)
        self.moduleUpsample4 = Upsample(256, 256)
        self.moduleUpsample3 = Upsample(128, 128)
        self.moduleUpsample2 = Upsample(64, 64)

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()
        #self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ])

    def forward(self, tensorFirst, tensorSecond):
        tensorJoin = torch.cat([tensorFirst, tensorSecond], 1) # 6  h
        tensorConv1 = self.moduleConv1(tensorJoin) # 6->32  h
        tensorPool1 = self.modulePool1(tensorConv1)# 32 h/2

        tensorConv2 = self.moduleConv2(tensorPool1) #32->64  h/2
        tensorPool2 = self.modulePool2(tensorConv2)# 64  h/4

        tensorConv3 = self.moduleConv3(tensorPool2)# 64->128  h/4
        tensorPool3 = self.modulePool3(tensorConv3)# 128  h/8

        tensorConv4 = self.moduleConv4(tensorPool3)# 128->256 h/8
        tensorPool4 = self.modulePool4(tensorConv4)#256 h/16

        tensorConv5 = self.moduleConv5(tensorPool4)#256->512 h/16
        tensorPool5 = self.modulePool5(tensorConv5)#512 h/32

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)#512->512 h/32
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)# 512 h/16

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)# 512->256 h/16
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)# 256 h/8

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)#256->128 h/8
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)# 128 h/4

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)#128->64 h/4
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)#64 h/2

        tensorCombine = tensorConv2 + tensorUpsample2

        vertical1 = self.moduleVertical1(tensorCombine)# 64 ->51 h
        horizontal1 = self.moduleHorizontal1(tensorCombine)
        vertical2 = self.moduleVertical2(tensorCombine)
        horizontal2 = self.moduleHorizontal2(tensorCombine)

        return torch.cat((vertical1,horizontal1,vertical2,horizontal2), 3)


# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, default =r'F:\xxx\githubitem\pytorch-sepconv\ckpt', help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, default =r'F:\xxx\pretrain_model\network-lf.pytorch',help='path of checkpoint for pretrained model')
parser.add_argument("--train_continue", type=bool, default=True, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=8, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=4, help='batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=100, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
args = parser.parse_args()

writer = SummaryWriter('log')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
moduleNetwork = Network()
moduleNetwork.to(device)

###Initializing VGG16 model for perceptual loss
vgg16 = torchvision.models.vgg16()
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)
for param in vgg16_conv_4_3.parameters():
		param.requires_grad = False

# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

trainset = dataloader.SepConvTrain(root=r'F:\xxx\data\txt\little_train.txt', transform=None, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)

validationset = dataloader.SepConvTrain(root=r'F:\xxx\data\txt\little_validation.txt', transform=None,randomCropSize=(640, 352), train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)

###Create transform to display image from tensor
negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

###Loss and Optimizer
L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()

params = list(moduleNetwork.parameters())
optimizer = optim.Adamax(params, lr=0.001, betas=(0.9, 0.999), eps=0.01, weight_decay=0)
# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)


def validate():
    # For details see training.
    psnr = 0
    tloss = 0
    flag = 1
    inum = 0
    icnt = random.randint(0, 10)
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData
            image1 = frame0.to(device)
            image3 = frame1.to(device)
            image2 = frameT.to(device)
            mypad = torch.nn.ReplicationPad2d(
                [int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)),
                 int(math.floor(51 / 2.0))])

            image1b = mypad(frame0).to(device)
            image3b = mypad(frame1).to(device)
            optimizer.zero_grad()

            # forward caluclation
            Kernel = moduleNetwork.forward(image1, image3)
            kernelDiv = torch.chunk(Kernel, 4, dim=3)
            tensorDot1 = sepconv.FunctionSepconv().forward(image1b, kernelDiv[0], kernelDiv[1]).detach()
            tensorDot2 = sepconv.FunctionSepconv().forward(image3b, kernelDiv[2], kernelDiv[3]).detach()
            tensorDot1.requires_grad = True
            tensorDot2.requires_grad = True
            tensorCombine = tensorDot1 + tensorDot2

            # backward caluclations
            recnLoss = L1_lossFn(tensorCombine, image2)
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(tensorCombine), vgg16_conv_4_3(image2))

            loss = 0.8 * recnLoss + 0.005 * prcpLoss
            tloss += loss.item()


            # For tensorboard
            if (flag and icnt == inum):
                idx = random.randint(0,frame0.size()[0] -1 )
                #f0 = pimg.fromarray((frame0[idx].numpy().transpose(1, 2, 0)[:, :, ::-1]* 255.0).astype(np.uint8))
                #f1 = pimg.fromarray((frame1[idx].numpy().transpose(1, 2, 0)[:, :, ::-1]* 255.0).astype(np.uint8))
                ft = pimg.fromarray((frameT[idx].numpy().transpose(1, 2, 0)[:, :, ::-1]* 255.0).astype(np.uint8))
                ftc = pimg.fromarray((tensorCombine.cpu()[idx].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8))
                alltrans = transforms.Compose([transforms.ToTensor()])
                #retImg = torchvision.utils.make_grid([alltrans(f0), alltrans(ft),alltrans(ftc),alltrans(f1)], padding=10)
                retImg = torchvision.utils.make_grid([alltrans(ft), alltrans(ftc)],padding=10)
                flag = 0

            # psnr
            MSE_val = MSE_LossFn(tensorCombine, image2)
            psnr += (10 * log10(1 / MSE_val.item()))
            inum +=1
            if inum > 10:
                break
    icnt = inum - 1
    return (psnr / icnt), (tloss / icnt), retImg
    #return (psnr / len(validationloader)), (tloss / len(validationloader)), retImg


if args.train_continue:
    dict1 = torch.load(args.checkpoint)
    moduleNetwork.load_state_dict(dict1)
    print("fine tuning from last!")

### Training
start = time.time()
checkpoint_counter = 0
### Main training loop
for epoch in range(args.epochs):
    print("Epoch: ", epoch)
    iLoss = 0

    # Increment scheduler count
    scheduler.step()
    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):

        ## Getting the input and the target from the training set
        frame0, frameT, frame1 = trainData
        image1 = frame0.to(device)
        image3 = frame1.to(device)
        image2 = frameT.to(device)
        mypad = torch.nn.ReplicationPad2d([int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)),
                                   int(math.floor(51 / 2.0))])

        image1b = mypad(frame0).to(device)
        image3b = mypad(frame1).to(device)


        # forward caluclation
        Kernel = moduleNetwork.forward(image1, image3)
        kernelDiv = torch.chunk(Kernel, 4, dim=3)
        tensorDot1 = sepconv.FunctionSepconv().forward(image1b, kernelDiv[0], kernelDiv[1]).detach()
        tensorDot2 = sepconv.FunctionSepconv().forward(image3b, kernelDiv[2], kernelDiv[3]).detach()
        tensorDot1.requires_grad = True
        tensorDot2.requires_grad = True
        tensorCombine = tensorDot1 + tensorDot2

        # backward caluclations
        recnLoss = L1_lossFn(tensorCombine, image2)
        prcpLoss = MSE_LossFn(vgg16_conv_4_3(tensorCombine), vgg16_conv_4_3(image2))

        loss = 0.8 * recnLoss +  0.005 * prcpLoss
        value_loss = loss.item()
        iLoss += value_loss
        loss.backward()
        kgrad1 = sepconv.FunctionSepconv().backward(tensorDot1.grad,(tensorDot1, image1b, kernelDiv[0], kernelDiv[1]))
        kgrad2 = sepconv.FunctionSepconv().backward(tensorDot2.grad,(tensorDot2, image3b, kernelDiv[2], kernelDiv[3]))
        kernelGrad = torch.cat((kgrad1[0], kgrad1[1], kgrad2[0], kgrad2[1]), 3)
        torch.autograd.backward([Kernel], [kernelGrad])

        optimizer.step()
        optimizer.zero_grad()
        #print("train epoch:" + str(epoch) + ",trainIndex:"+ str(trainIndex) +  ",value_loss:" + str(value_loss))

        # Validation and progress every `args.progress_iter` iterations
        if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
            end = time.time()
            psnr, vLoss, valImg = validate()

            # Tensorboard
            itr = trainIndex + epoch * (len(trainloader))
            writer.add_scalars('Loss', {'trainLoss': iLoss / args.progress_iter,
                                        'validationLoss': vLoss}, itr)
            writer.add_scalar('PSNR', psnr, itr)

            writer.add_image('Validation', valImg, itr)

            endVal = time.time()
            print(
                " Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (
                iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr, endVal - end,
                get_lr(optimizer)))
            iLoss = 0
            start = time.time()
            #######Save Models#######
            if (epoch % args.checkpoint_epoch == 0):
                torch.save(moduleNetwork.state_dict(), args.checkpoint_dir + "/SepConv" + str(checkpoint_counter) + ".pytorch")
                checkpoint_counter += 1

print('process finished')

