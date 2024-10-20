import os
import time
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from model import CLIPDeRouter
from PIL import Image
import random
from torch.utils.data import Dataset
import math
from utils_image import process_highres_image
from tqdm import tqdm
from torch.cuda.amp import autocast

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        help="path to input images",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to ckpt",
    )
    parser.add_argument(
        "--label",
        type=int,
        help="label",
    )
    
    args = parser.parse_args()
    return args

class AllWeather_Dataset_cls_test(Dataset):
    def __init__(self, path, label,
                processor=None):
        files = os.listdir(path)
        self.imgs = [os.path.join(path, k) for k in files]
        self.label = label

        self.processor = processor

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        data = process_highres_image(Image.open(self.imgs[idx]).convert('RGB'),self.processor,self.processor.size["shortest_edge"]*2)

        return data, self.label

if __name__ == '__main__':
    args = prepare()
    cls_num = 3
    model = CLIPDeRouter("openai/clip-vit-large-patch14-336",  cls_num)
    model.to(dtype=torch.bfloat16)
    image_processor = model.clip_model.image_processor
    ckpt = torch.load(args.ckpt, map_location='cpu')['params']

    model.load_state_dict(ckpt)
    model.eval()
    model = model.cuda()

    dataset_test = AllWeather_Dataset_cls_test(args.path,args.label,image_processor)
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    size = torch.tensor(0.0).cuda()
    correct = torch.tensor(0.0).cuda()
    for i, (images, labels) in enumerate(tqdm(dataloader_test)):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                outputs = model(images)[0]
            size += images.size(0)
        correct += (outputs.argmax(1) == labels).type(torch.float).sum()
        acc = correct / size
        print(f"{i}, Acc is {acc:.2%}")

    acc = correct / size
    print(f'Toal num is {size}')
    print(f'Correct num is {correct}')
    print(f"Accuracy is {acc:.2%}")
