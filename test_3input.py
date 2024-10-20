import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from natsort import natsorted
from glob import glob
from basicsr.models.archs.PriorFtModel_arch import PriorFtModelL
from skimage import img_as_ubyte
from pdb import set_trace as stx
import cv2
from classification.model import CLIPDeRouter
from copy import deepcopy
from PIL import Image
from classification.utils_image import process_highres_image
from torch.cuda.amp import autocast
from basicsr.utils import FileClient, imfrombytes,img2tensor


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

parser = argparse.ArgumentParser(description='Inference')

parser.add_argument('--input_dir', default='/path/to/input', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/path/to/weight', type=str, help='Path to weights')
parser.add_argument('--weights_clip', default='/path/to/clip', type=str, help='Path to clip weights')
parser.add_argument('--yaml_file', default='./path/to/yaml', type=str, help='Path to yaml file')

args = parser.parse_args()

####### Load yaml #######
yaml_file = args.yaml_file
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
model_restoration = PriorFtModelL(**x['network_g'])

file_client= FileClient('disk')

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration.eval()

opt_clip = deepcopy(x['network_clip'])
net_clip = CLIPDeRouter(opt_clip.pop('vision_tower'), opt_clip.pop('cls_num'))
net_clip.to(dtype=torch.bfloat16)
image_processor = net_clip.clip_model.image_processor
ckpt = torch.load(args.weights_clip, map_location='cpu')['params']
net_clip.load_state_dict(ckpt)
net_clip.cuda()
net_clip.eval()
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

inp_dir = args.input_dir
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img_bytes = file_client.get(file_, 'lq')
        input_ = imfrombytes(img_bytes, float32=True)
        input_ = img2tensor(input_, bgr2rgb=True, float32=True)
        input_ = input_.unsqueeze(0).cuda()

        img_bytes = file_client.get(file_, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        img_clip = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
        img_clip = (img_clip * 255).astype(np.uint8)
        img_clip = Image.fromarray(img_clip)
        img_clip = process_highres_image(img_clip, image_processor, image_processor.size["shortest_edge"] * 2)
        img_clip = img_clip.unsqueeze(0).cuda()

        with autocast(dtype=torch.bfloat16):
            probs, de_prior = net_clip(img_clip)
            probs = probs.float()
            de_prior = de_prior.float()

        restored = model_restoration(input_, de_prior, probs)

        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        save_img(os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png'), img_as_ubyte(restored))

