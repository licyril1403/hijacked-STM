from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torchvision.transforms.functional as TF

# general libs
from PIL import Image
from PIL import ImageDraw
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import random


### My libs
from model import STM, set_masks
from helpers import overlay_davis


torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="Hijacked STM")
    parser.add_argument("-ann", type=str, help="Path to the annotation image", required=True)
    parser.add_argument("-img", type=str, help="Path to the slice image", required=True)
    parser.add_argument("-vol", type=str, help="Path to the volume folder", required=True)
    parser.add_argument("-F", help="Put the pseudo labels of the annotation slice in the memery", action="store_true")
    parser.add_argument("-STM", help="Run with default STM", action="store_true")

    return parser.parse_args()

args = get_arguments()

WEIGHTS_PATH = "STM_weights.pth"
PROPAG = args.F
MASK = not args.STM
ANN = args.ann
IMG = args.img
VOL = args.vol


MODEL = 'Hi-STM'
if not MASK:
    MODEL = 'STM'

print(MODEL, ': Testing on', VOL, 'with', ANN)

code = MODEL

if torch.cuda.is_available():
    print('Using Cuda devices, num:', torch.cuda.device_count())
else:
    print('No cuda. Exit.')
    exit()


if torch.cuda.device_count() == 0:
    print('No GPU detected. Exit.')
    exit()

if PROPAG:
    print('--- Using pseudo labels in memory')
    code += '_F'

set_masks(MASK)

test_path = os.path.join(VOL, '..', 'test', code)
viz_path = os.path.join(VOL, '..','viz', code)

if not os.path.exists(test_path):
    os.makedirs(test_path)

if not os.path.exists(viz_path):
    os.makedirs(viz_path)


# Loading model

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = WEIGHTS_PATH
print('Loading weights:', pth_path)
dic=torch.load(pth_path)
model.load_state_dict(dic)

NOBJ = 1

# Loading images

allfiles = os.listdir(VOL)
allfiles.sort()

N_frames = []
names = []


for f in allfiles:
    if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"):
        img_file = VOL + '/' + f  
        names.append(f)
        N_frames.append(np.array(Image.open(img_file).convert('RGB'))/255.)

n_images = len(N_frames)

N_frames = np.asarray([N_frames])
Fs = torch.from_numpy(np.transpose(N_frames.copy(), (0, 4, 1, 2, 3)).copy()).float()

# Loading annotations

A = np.array(Image.open(ANN).convert('RGB'))/255.
I = np.array(Image.open(IMG).convert('RGB'))/255.

Inp = np.zeros((1, 3, I.shape[0], I.shape[1]))
Mnp = np.zeros((1, 11, I.shape[0], I.shape[1]))
Ac = np.zeros((1, 1, 1, int(I.shape[0]/8), int(I.shape[1]/8)))

Inp[0, 0] = I[:, :, 0]
Inp[0, 1] = I[:, :, 1]
Inp[0, 2] = I[:, :, 2]


Mnp[0, 0] = A[:,:,0]
Mnp[0, 1] = A[:,:,1]

Am = A[:,:,2]
Am = 1 - Am

A = torch.from_numpy(Am).float()
A = A.view(1, 1, A.size(0), A.size(1))
A = F.interpolate(A, scale_factor=1/8, mode='bilinear', align_corners=False)

A = A.view(A.size(2), A.size(3))
Ac[:, :, 0] = A

Ac = torch.from_numpy(Ac).float()
Fc = torch.from_numpy(Inp).float()
Mc = torch.from_numpy(Mnp).float()

with torch.no_grad():
    key, value = model(Fc, Mc, torch.tensor([NOBJ])) 


if PROPAG:
    with torch.no_grad():
        logit = model(Fc, key, value, torch.tensor([NOBJ]), an_mask=Ac) 
        Es = F.softmax(logit, dim=1)
        pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)


        Mnp[0, 0] = pred == 0
        Mnp[0, 1] = pred == 1

        Mc = torch.from_numpy(Mnp).float()

        key, value = model(Fc, Mc, torch.tensor([NOBJ])) 

    Ac = None


# Propagating into the entire volume

print('Start Testing:', code)
print('[{}]: num_frames: {}, num_objects: {}'.format(VOL, n_images, NOBJ))


for t in range(n_images):
    print('Processing', names[t])
    with torch.no_grad():
        logit = model(Fs[:,:,t], key, value, torch.tensor([NOBJ]), an_mask=Ac)
    Es = F.softmax(logit, dim=1)

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    pF = (Fs[0,:,t].permute(1,2,0).numpy() * 255.).astype(np.uint8)
    pE = (pred==1)*255

    img_E = Image.fromarray(pE.astype(np.uint8))
    img_E.save(os.path.join(test_path, names[t]))

    canvas = overlay_davis(pF, pred)
    canvas = Image.fromarray(canvas)
    canvas.save(os.path.join(viz_path, names[t]))

