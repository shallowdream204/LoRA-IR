# A simple training code, directly modified from https://github.com/rickyang1114/DDP-practice/blob/main/ddp_main.py
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


def augment_image(image):
    angle = random.choice([0, 90, 180, 270])
    image = image.rotate(angle, expand=True)

    # 随机选择是否翻转
    if random.choice([True, False]):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    return image

class AllWeather_Dataset_cls_train(Dataset):
    def __init__(self, root0_in, root1_in, root2_in,
                 fix_sample, processor=None):
        
        fix_sample_0 = fix_sample
        in_files_0 = os.listdir(root0_in)
        if fix_sample_0 > len(in_files_0):
            fix_sample_0 = len(in_files_0)
        in_files_0 = random.sample(in_files_0, fix_sample_0)   
        self.imgs_in_0 = [os.path.join(root0_in, k) for k in in_files_0]
        len_imgs_in_0 = len(self.imgs_in_0)

        def sample_and_match(root_in, fix_sample, len_imgs_in_0):
            in_files = os.listdir(root_in)
            if fix_sample > len(in_files):
                fix_sample = len(in_files)
            in_files = random.sample(in_files, fix_sample)
            imgs_in = [os.path.join(root_in, k) for k in in_files]
            len_imgs_in_ori = len(imgs_in)
            imgs_in = imgs_in * (math.ceil(len_imgs_in_0 / len_imgs_in_ori))
            imgs_in = imgs_in[0: len_imgs_in_0]
            return imgs_in

        self.imgs_in_1 = sample_and_match(root1_in, fix_sample, len_imgs_in_0)
        self.imgs_in_2 = sample_and_match(root2_in, fix_sample, len_imgs_in_0)

        self.processor = processor

    def __len__(self):
        return len(self.imgs_in_0)

    def __getitem__(self, idx):
        data_0 = process_highres_image(augment_image(Image.open(self.imgs_in_0[idx]).convert('RGB')), self.processor, self.processor.size["shortest_edge"] * 2)
        data_1 = process_highres_image(augment_image(Image.open(self.imgs_in_1[idx]).convert('RGB')), self.processor, self.processor.size["shortest_edge"] * 2)
        data_2 = process_highres_image(augment_image(Image.open(self.imgs_in_2[idx]).convert('RGB')), self.processor, self.processor.size["shortest_edge"] * 2)

        return [data_0, 0], [data_1, 1], [data_2, 2]


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=8,
        type=int,
        metavar="N",
        help="number of batchsize",
    )
    parser.add_argument(
        "--path_snow",
        default='/path/to/snow',
        type=str,
        help="path to snow trainset",
    )
    parser.add_argument(
        "--path_rain",
        default='/path/to/rain',
        type=str,
        help="path to rain trainset",
    )
    parser.add_argument(
        "--path_raindrop",
        default='/path/to/raindrop',
        type=str,
        help="path to raindrop trainset",
    )
    args = parser.parse_args()

    # The following environment variables are set to enable DDP
    os.environ["MASTER_ADDR"] = "localhost"  # IP address of the master machine
    os.environ["MASTER_PORT"] = "1234"  # port number of the master machine
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # specify the GPUs to use
    world_size = torch.cuda.device_count()
    os.environ["WORLD_SIZE"] = str(world_size)
    return args


def init_ddp(local_rank):
    # after this setup, tensors can be moved to GPU via `a = a.cuda()` rather than `a = a.to(local_rank)`
    torch.cuda.set_device(local_rank)
    os.environ["RANK"] = str(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def get_bare_model(net):
    if isinstance(net, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        net = net.module
    return net

def train(model, train_dloader, criterion, optimizer, epoch):
    local_rank = dist.get_rank()
    running_loss = 0.0
    model.train()
    for i, train_data in enumerate(tqdm(train_dloader)):
        optimizer.zero_grad()
        data_A, data_B, data_C = train_data
        images = torch.cat([data_A[0],data_B[0],data_C[0]],dim=0).cuda()
        labels = torch.cat([data_A[1],data_B[1],data_C[1]],dim=0).cuda()
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)[0]
            loss = criterion(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()  ###
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:    
            if local_rank == 0:
                current_lr = optimizer.param_groups[0]['lr'] 
                print('[%d, %5d] loss: %.5f, lr: %.6f' %
                  (epoch + 1, i + 1, running_loss / 10, current_lr))
            running_loss = 0.0
    if local_rank == 0:
        save_dict = {}
        net_ = get_bare_model(model)
        state_dict = net_.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            state_dict[key] = param.cpu()
        save_dict['params'] = state_dict
        torch.save(save_dict, f"checkpoint_epoch_{epoch+1}.pth")

def main(local_rank, args):
    init_ddp(local_rank)  ### init DDP
    model = (
        CLIPDeRouter("openai/clip-vit-large-patch14-336", cls_num=3).cuda()
    )
    model.to(dtype=torch.bfloat16)
      ### Note: the `forward` method of the model has been modified
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  ### Convert BatchNorm layers
    image_processor = model.clip_model.image_processor
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )  ### Wrap with DDP
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    train_dataset = AllWeather_Dataset_cls_train(args.path_snow,
                                     args.path_rain,
                                     args.path_raindrop,10000,
                                     image_processor)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset
    )  ### Sampler specifically for DDP
    g = get_ddp_generator()  ###
    train_dloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  ### shuffle is mutually exclusive with sampler
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
        generator=g,
        drop_last=True
    )  ### generator is used for random seed
    
    for epoch in range(args.epochs):
        if local_rank == 0:  ### avoid redundant printing for each process
            print(f"begin training of epoch {epoch + 1}/{args.epochs}")
        train_dloader.sampler.set_epoch(epoch)  ### set epoch for sampler
        train(model, train_dloader, criterion, optimizer, epoch)
        scheduler.step()

    dist.destroy_process_group() ### destroy the process group, in accordance with init_process_group.


if __name__ == "__main__":
    args = prepare()
    time_start = time.time()
    mp.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print(f"\ntime elapsed: {time_elapsed:.2f} seconds")