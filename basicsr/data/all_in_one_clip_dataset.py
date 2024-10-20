# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import cv2
import numpy as np
from PIL import Image
from classification.utils_image import process_highres_image
from transformers import CLIPImageProcessor
import os
import math
import random

class AllInOneCLIPDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(AllInOneCLIPDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        self.gt_folders, self.lq_folders = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # if self.io_backend_opt['type'] == 'lmdb':
        #     self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
        #     self.io_backend_opt['client_keys'] = ['lq', 'gt']
        #     self.paths = paired_paths_from_lmdb(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt[
        #         'meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.opt['meta_info_file'], self.filename_tmpl)
        # else:
        #     self.paths = paired_paths_from_folder(
        #         [self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #         self.filename_tmpl)
        self.paths_lq = []
        self.paths_gt = []
        self.num_tasks = len(self.lq_folders)
        for i in range(self.num_tasks):
            lq_imgs = os.listdir(self.lq_folders[i])
            lq_paths = [os.path.join(self.lq_folders[i],lq_img) for lq_img in lq_imgs]
            gt_paths = [os.path.join(self.gt_folders[i],lq_img) for lq_img in lq_imgs]
            self.paths_lq.append(lq_paths)
            self.paths_gt.append(gt_paths)
        paths_len = [len(x) for x in self.paths_lq]
        self.max_task_num = max(paths_len)
        for i in range(self.num_tasks):
            len_ori = len(self.paths_lq[i])
            self.paths_lq[i] = self.paths_lq[i] * (math.ceil(self.max_task_num / len_ori))
            self.paths_lq[i]  = self.paths_lq[i][0: self.max_task_num]
            self.paths_gt[i] = self.paths_gt[i] * (math.ceil(self.max_task_num / len_ori))
            self.paths_gt[i]  = self.paths_gt[i][0: self.max_task_num]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        i = random.randint(0,self.num_tasks-1)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths_gt[i][index]
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths_lq[i][index]
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        if self.opt['phase'] == 'val':
            img_clip = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
            img_clip = (img_clip * 255).astype(np.uint8)
            img_clip = Image.fromarray(img_clip)
            img_clip = process_highres_image(img_clip,self.processor,self.processor.size["shortest_edge"]*2)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

            # TODO: color space transform
            # BGR to RGB, HWC to CHW, numpy to tensor
            img_clip = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
            img_clip = (img_clip * 255).astype(np.uint8)
            img_clip = Image.fromarray(img_clip)
            img_clip = process_highres_image(img_clip,self.processor,self.processor.size["shortest_edge"]*2)

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)


        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_clip': img_clip,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return self.max_task_num