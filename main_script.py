from ngransac_demo import matcher
import sys
#project_name = 'craft'
#sys.path.append(project_name)
#sys.path.append('craft/basenet')
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft.craft_utils
import craft.imgproc
# import craft.file_utils
import json
import zipfile

from craft.craft import CRAFT
from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = craft.imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = craft.imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft.craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = craft.imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


img1_pth = './images/demo1.jpg'
img2_pth = './images/demo2.jpg'
pretrained_model = './models/craft_mlt_25k.pth'
inlier_mask, all_matches, kp1, kp2 = matcher(img1_pth, img2_pth, 'matching_op.png', 900, 900, '', 1000, './models/weights_e2e_E_r1.00_.net', False)

good_matches = []
for i in range(len(all_matches)):
  if inlier_mask[i] == 1:
    good_matches.append(all_matches[i])

point_corresp = []
for match in good_matches:
  p1 = kp1[match.queryIdx].pt
  p2 = kp2[match.trainIdx].pt
  point_corresp.append((p1,p2))
  
  
net = CRAFT()     # initialize

# print('Loading weights from checkpoint (' + a + ')')
# if args.cuda:
net.load_state_dict(copyStateDict(torch.load(pretrained_model)))
# else:
#     net.load_state_dict(copyStateDict(torch.load(model, map_location='cpu')))

# if args.cuda:
net = net.cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = False

net.eval()


# t = time.time()

  # load data

import cv2
image1 = cv2.imread(img1_pth)
print(image1.shape)
image1 = cv2.resize(image1, ((int(image1.shape[1]*0.2)), (int(image1.shape[0]*0.2))), interpolation = cv2.INTER_AREA)
print(image1.shape)
# cv2.imwrite(img1.split('.')[0] + 'resized.' + img1.split('.')[1], i1)

image2 = cv2.imread(img2_pth)
print(image2.shape)
image2 = cv2.resize(image2, ((int(image2.shape[1]*0.2)), (int(image2.shape[0]*0.2))), interpolation = cv2.INTER_AREA)
# cv2.imwrite(img2.split('.')[0] + 'resized.' + img2.split('.')[1], i2)
print(image2.shape)
# img1 = img1.split('.')[0] + 'resized.' + img1.split('.')[1]
# img2 = img2.split('.')[0] + 'resized.' + img2.split('.')[1]

# image1 = imgproc.loadImage(img1)
# image2 = imgproc.loadImage(img2)

bboxes1, polys1, score_text1 = test_net(net, image1, 0.4, 0.4, 0.4, True, False, None)
bboxes2, polys2, score_text2 = test_net(net, image2, 0.4, 0.4, 0.4, True, False, None)

  



