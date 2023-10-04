import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
import sys


def Normalization255_GRAY(img, max_value=255, min_value=0):
    Max = np.max(img)
    Min = np.min(img)
    img = ((img - Min) / (Max - Min)) * (max_value - min_value) + min_value
    return img

def filter_nn(img, kernel, padding=1):
    # expected img:   FloatTensor [w,h]
    # expected kernel:  FloatTensor [size,size]
    img = torch.Tensor(img)
    img = img.unsqueeze(0).unsqueeze(0)
    img = img.float()
    kernel = torch.Tensor(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    res = F.conv2d(img, weight, padding=padding)
    return res

def produce_x(img_gray, R=1):
    # produce x component,
    # require:img_gray as narray
    # return: img_x as tensor
    if R == 1:
        filter_x = np.array([[-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))],
                             [-1, 0, 1],
                             [-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))]])

    if R == 2:
        filter_x = np.array([
            [np.cos(6 / 8 * np.pi) / 8, np.cos(5 / 8 * np.pi) / 5, np.cos(4 / 8 * np.pi) / 4,
             np.cos(3 / 8 * np.pi) / 5, np.cos(2 / 8 * np.pi) / 8],
            [np.cos(7 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(1 / 8 * np.pi) / 5],
            [np.cos(8 / 8 * np.pi) / 4, -1, 0, 1, np.cos(0 / 8 * np.pi) / 4],
            [np.cos(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(15 / 8 * np.pi) / 5],
            [np.cos(10 / 8 * np.pi) / 8, np.cos(11 / 8 * np.pi) / 5, np.cos(12 / 8 * np.pi) / 4,
             np.cos(13 / 8 * np.pi) / 5,
             np.cos(14 / 8 * np.pi) / 8]])
    if R == 3:
        filter_x = np.array([[np.cos(9 / 12 * np.pi)/18,np.cos(8 / 12 * np.pi)/13,np.cos(7 / 12 * np.pi)/10,np.cos(6 / 12 * np.pi)/9,
                              np.cos(5 / 12 * np.pi) / 10,np.cos(4 / 12 * np.pi)/13,np.cos(3 / 12 * np.pi)/18],
            [np.cos(10 / 12 * np.pi)/13,np.cos(6 / 8 * np.pi) / 8, np.cos(5 / 8 * np.pi) / 5, np.cos(4 / 8 * np.pi) / 4,
             np.cos(3 / 8 * np.pi) / 5, np.cos(2 / 8 * np.pi) / 8,np.cos(2 / 12 * np.pi)/13],
            [np.cos(11 / 12 * np.pi)/10,np.cos(7 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(1 / 8 * np.pi) / 5,np.cos(1 / 12 * np.pi)/10],
            [np.cos(12 / 12 * np.pi)/9,np.cos(8 / 8 * np.pi) / 4, -1, 0, 1, np.cos(0 / 8 * np.pi) / 4,np.cos(0 / 12 * np.pi)/9],
            [np.cos(13 / 12 * np.pi)/10,np.cos(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(15 / 8 * np.pi) / 5,np.cos(23 / 12 * np.pi)/10],
            [np.cos(14 / 12 * np.pi)/13,np.cos(10 / 8 * np.pi) / 8, np.cos(11 / 8 * np.pi) / 5, np.cos(12 / 8 * np.pi) / 4,
             np.cos(13 / 8 * np.pi) / 5,np.cos(14 / 8 * np.pi) / 8,np.cos(22 / 12 * np.pi)/13],
            [np.cos(15 / 12 * np.pi)/18,np.cos(16 / 12 * np.pi)/13,np.cos(17 / 12 * np.pi)/10,np.cos(18 / 12 * np.pi)/9,
             np.cos(19 / 12 * np.pi) / 10,np.cos(20 / 12 * np.pi)/13,np.cos(21 / 12 * np.pi)/18]])
    img_x = filter_nn(img_gray, filter_x, padding=R)
    img_xorl = np.array(img_x).reshape(img_x.shape[2], -1)

    return img_xorl

def produce_y(img_gray, R=1):
    # produce x component,
    # require:img_gray as narray
    # return: img_x as tensor
    if R == 1:
        filter_y = np.array([[1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2))],
                             [0, 0, 0],
                             [-1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2))]])

    if R == 2:
        filter_y = np.array([
            [np.sin(6 / 8 * np.pi) / 8, np.sin(5 / 8 * np.pi) / 5, np.sin(4 / 8 * np.pi) / 4,
             np.sin(3 / 8 * np.pi) / 5, np.sin(2 / 8 * np.pi) / 8],
            [np.sin(7 / 8 * np.pi) / 5, 1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2)), np.sin(1 / 8 * np.pi) / 5],
            [np.sin(8 / 8 * np.pi) / 4, 0, 0, 0, np.sin(0 / 8 * np.pi) / 4],
            [np.sin(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2)), np.sin(15 / 8 * np.pi) / 5],
            [np.sin(10 / 8 * np.pi) / 8, np.sin(11 / 8 * np.pi) / 5, np.sin(12 / 8 * np.pi) / 4,
             np.sin(13 / 8 * np.pi) / 5,
             np.sin(14 / 8 * np.pi) / 8]])
    if R == 3:
        filter_y = np.array([[np.sin(9 / 12 * np.pi) / 18, np.sin(8 / 12 * np.pi) / 13, np.sin(7 / 12 * np.pi) / 10,
                              np.sin(6 / 12 * np.pi) / 9,
                              np.sin(5 / 12 * np.pi) / 10, np.sin(4 / 12 * np.pi) / 13, np.sin(3 / 12 * np.pi) / 18],
                             [np.sin(10 / 12 * np.pi) / 13, np.sin(6 / 8 * np.pi) / 8, np.sin(5 / 8 * np.pi) / 5,
                              np.sin(4 / 8 * np.pi) / 4,
                              np.sin(3 / 8 * np.pi) / 5, np.sin(2 / 8 * np.pi) / 8, np.sin(2 / 12 * np.pi) / 13],
                             [np.sin(11 / 12 * np.pi) / 10, np.sin(7 / 8 * np.pi) / 5, 1 / (2 * np.sqrt(2)), 1,
                              1 / (2 * np.sqrt(2)), np.sin(1 / 8 * np.pi) / 5, np.sin(1 / 12 * np.pi) / 10],
                             [np.sin(12 / 12 * np.pi) / 9, np.sin(8 / 8 * np.pi) / 4, 0, 0, 0,
                              np.sin(0 / 8 * np.pi) / 4, np.sin(0 / 12 * np.pi) / 9],
                             [np.sin(13 / 12 * np.pi) / 10, np.sin(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), -1,
                              -1 / (2 * np.sqrt(2)), np.sin(15 / 8 * np.pi) / 5, np.sin(23 / 12 * np.pi) / 10],
                             [np.sin(14 / 12 * np.pi) / 13, np.sin(10 / 8 * np.pi) / 8, np.sin(11 / 8 * np.pi) / 5,
                              np.sin(12 / 8 * np.pi) / 4,
                              np.sin(13 / 8 * np.pi) / 5, np.sin(14 / 8 * np.pi) / 8, np.sin(22 / 12 * np.pi) / 13],
                             [np.sin(15 / 12 * np.pi) / 18, np.sin(16 / 12 * np.pi) / 13,
                              np.sin(17 / 12 * np.pi) / 10, np.sin(18 / 12 * np.pi) / 9,
                              np.sin(19 / 12 * np.pi) / 10, np.sin(20 / 12 * np.pi) / 13,
                              np.sin(21 / 12 * np.pi) / 18]])

    img_y = filter_nn(img_gray, filter_y, padding=R)
    img_yorl = np.array(img_y).reshape(img_y.shape[2], -1)

    return img_yorl

def filter_8_1(img):
    img = np.where(img > 2, img, 2)
    img_xorl = produce_x(img, 1)
    img_yorl = produce_y(img, 1)
    magtitude = np.arctan(np.sqrt((np.divide(img_xorl, img + 0.0001) ** 2) + (np.divide(img_yorl, img + 0.0001) ** 2)))
    magtitude = Normalization255_GRAY(magtitude, 255, 1)
    return magtitude

def filter_16_2(img):
    img = np.where(img > 2, img, 2)
    img_xorl = produce_x(img, 2)
    img_yorl = produce_y(img, 2)
    magtitude = np.arctan(np.sqrt((np.divide(img_xorl, img + 0.0001) ** 2) + (np.divide(img_yorl, img + 0.0001) ** 2)))
    magtitude = Normalization255_GRAY(magtitude, 255, 1)
    return magtitude

def filter_24_3(img):
    img = np.where(img > 2, img, 2)
    img_xorl = produce_x(img, 3)
    img_yorl = produce_y(img, 3)
    magtitude = np.arctan(np.sqrt((np.divide(img_xorl, img + 0.0001) ** 2) + (np.divide(img_yorl, img + 0.0001) ** 2)))
    magtitude = Normalization255_GRAY(magtitude, 255, 1)
    return magtitude

def plgf(I, R=3):
    if R == 1:
        I = filter_8_1(I)
    if R == 2:
        I = filter_16_2(I)
    if R == 3:
        I = filter_24_3(I)
    out = I
    return out

if __name__ == '__main__':
    im = cv2.imread('example.jpg')
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    p = plgf(im_gray)



    cv2.imwrite('p.jpg', p.astype(np.uint8))

    p = plgf(im[:,:,0])
    cv2.imwrite('plgf_b.jpg', p.astype(np.uint8))

    p = plgf(im[:, :, 1])
    cv2.imwrite('plgf_g.jpg', p.astype(np.uint8))

    p = plgf(im[:, :, 2])
    cv2.imwrite('plgf_r.jpg', p.astype(np.uint8))

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    p = plgf(im_hsv[:, :, 0])
    cv2.imwrite('plgf_h.jpg', p.astype(np.uint8))

    p = plgf(im_hsv[:, :, 1])
    cv2.imwrite('plgf_s.jpg', p.astype(np.uint8))

    p = plgf(im_hsv[:, :, 2])
    cv2.imwrite('plgf_v.jpg', p.astype(np.uint8))

    import IPython; IPython.embed()