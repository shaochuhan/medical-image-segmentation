import argparse
import json
import logging
import sys

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from UNet import Unet
from Attention_UNet import AttU_Net

from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot
from torchvision.models import vgg16

from PIL import Image
import cv2

from utils.colors import get_colors

import argparse
import logging
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import pickle

from utils.dataset import BasicDataset


def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=21)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='Attention_UNet',
                       help='UNet/Attention_UNet/')
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument('--dataset', default='Prostate',
                       help='dataset name:liver/')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    args = parse.parse_args()
    return args


def getLog(args):
    dirname = os.path.join(args.log_dir, args.arch, str(args.batch_size), str(args.dataset), str(args.epoch))
    filename = dirname + '/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logging


def getModel(args):
    if args.arch == 'UNet':
        model = Unet(3, 1).to(device)

    if args.arch == 'Attention_UNet':
        model = AttU_Net(3, 1).to(device)

    return model


def getDataset(args):
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    if args.dataset == 'Prostate':
        train_dataset = ProstateDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = ProstateDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    return train_dataloaders, val_dataloaders, test_dataloaders


def val(model, best_iou, val_dataloaders):
    model = model.eval()
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  # 验证集图片的总数
        # print(num)
        for x, _, pic, mask in val_dataloaders:
            x = x.to(device)
            y, feature = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0], img_y)  # 获取当前预测图的miou，并加到总miou中
            dice_total += get_dice(mask[0], img_y)
            if i < num: i += 1  # 处理验证集下一张图
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total / num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou, aver_hd, aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou, aver_hd, aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(
                args.dataset) + '_' + str(args.epoch) + '.pth')
        return best_iou, aver_iou, aver_dice, aver_hd


def train(model, criterion, optimizer, train_dataloader, val_dataloader, args):
    best_iou, aver_iou, aver_dice, aver_hd = -1, 0, 0, 0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        features = {}
        num = 0
        for x, y, _, mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    # print(output.size())
                    output1 = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
                    # print(labels.size())
                    loss += criterion(output1, labels)
                loss /= len(outputs)
            else:
                output, feature = model(inputs)
                name = '{:03d}.png'.format(num)
                feature = feature.cpu()
                # 将张量转换为NumPy数组
                array = feature.detach().numpy()
                # print(array.shape)
                features[name] = array
                directory = os.path.join(r'result/feature', str(args.arch), str(args.batch_size))
                # 定义要创建的txt文件路径以及文件名
                filename = directory + '/{:02d}_feature.txt'.format(epoch)
                # 如果文件夹不存在就进行创建
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(directory + '/{:02d}_feature.txt'.format(epoch), 'a') as f:
                    # 附加值的标签
                    f.write('{}:\n'.format(name))
                    # 保存数组到文本文件中
                    np.savetxt(f, array.reshape(1, -1), delimiter='\t', fmt='%d')
                    # 附加一个空行
                    f.write('\n')
                num = num + 1
                output1 = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
                loss = criterion(output1, labels)
            if threshold != None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info(
                "%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_iou, aver_iou, aver_dice, aver_hd = val(model, best_iou, val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice', iou_list, dice_list)
    metrics_plot(args, 'hd', hd_list)
    return model


def test(val_dataloaders, save_predict=False):
    logging.info('final test........')
    if save_predict == True:
        dir = os.path.join(r'./saved_predict', str(args.arch), str(args.batch_size), str(args.epoch), str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(
        r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(
            args.epoch) + '.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)  # 验证集图片的总数
        for pic, _, pic_path, mask_path in val_dataloaders:
            pic = pic.to(device)

            predict, feature = model(pic)
            feature = feature.cpu()
            # 将张量转换为NumPy数组
            array = feature.detach().numpy()
            # print(array.shape)
            with open('result/feature_list.txt', 'a') as f:
                np.savetxt(f, array.reshape(1, -1), delimiter='\t', fmt='%d')# 附加一个空行
                f.write('\n')

            # if args.deepsupervision:
            #     predict = torch.squeeze(predict[-1]).cpu().numpy()
            # else:
            #     predict = torch.squeeze(predict).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #
            # iou = get_iou(mask_path[0], predict)
            # miou_total += iou  # 获取当前预测图的miou，并加到总miou中
            # hd_total += get_hd(mask_path[0], predict)
            # dice = get_dice(mask_path[0], predict)
            # dice_total += dice
            #
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 3, 1)
            # ax1.set_title('input')
            # plt.imshow(Image.open(pic_path[0]))
            # # print(pic_path[0])
            # ax2 = fig.add_subplot(1, 3, 2)
            # ax2.set_title('predict')
            # plt.imshow(predict, cmap='Greys_r')
            # ax3 = fig.add_subplot(1, 3, 3)
            # ax3.set_title('mask')
            # plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            # if save_predict == True:
            #     plt.savefig(dir + '/' + mask_path[0].split('/')[-1])
            # print('iou={},dice={}'.format(iou, dice))
            if i < num: i += 1  # 处理验证集下一张图

        # 定义要创建的txt文件路径以及文件名
        # 如果文件夹不存在就进行创建
        # for i in feature_list:
        #     with open('result/feature_list.txt', 'a') as f:
        #         np.savetxt(f, feature_list[i].reshape(1, -1), delimiter='\t', fmt='%d')# 附加一个空行
        #         f.write('\n')

        # print('Miou=%f,aver_hd=%f,dv=%f' % (miou_total / num, hd_total / num, dice_total / num))
        # logging.info('Miou=%f,aver_hd=%f,dv=%f' % (miou_total / num, hd_total / num, dice_total / num))


if __name__ == "__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size, args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
                 (args.arch, args.epoch, args.batch_size, args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders, val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)
