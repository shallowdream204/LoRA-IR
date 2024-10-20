import argparse
import cv2
import os
import numpy as np
from skimage.metrics import mean_squared_error
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
#import lpips
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

#from niqe.niqe import compute_niqe

#criterion = lpips.LPIPS(net='vgg', lpips=True, pnet_rand=False, pretrained=True).cuda()
def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    :parame img: uint8 or float ndarray
    '''
    in_im_type = im.dtype
    im = im.astype(np.float64)
    if in_im_type != np.uint8:
        im *= 255.
    # convert
    if only_y:
        rlt = np.dot(im, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im, np.array([[65.481,  -37.797, 112.0  ],
                                      [128.553, -74.203, -93.786],
                                      [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if in_im_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_im_type)

def rgb2ycbcrTorch(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C, [0,255]
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

def readim(file):
    # print(file)
    img = cv2.imread(file)
    img = img.astype(np.float32)
    return img / 255.

def loadfiles(folder):
    files = os.listdir(folder)
    return natsorted(files)

def resize(im, size, crop=True):
    if crop:
        return im[:size[0], :size[1]]
    else:
        return cv2.resize(im, size)

from natsort import natsorted

def np2torch(img):
    im = img.astype(np.float32) / 255
    im = torch.tensor(im).permute((2,0,1)).unsqueeze(0)
    return im.cuda()

def compute_metrics_for_pair(file1, file2, path1, path2, ycbcr=True):
    img1 = readim(os.path.join(path1, file1))
    img2 = readim(os.path.join(path2, file2))
    
    if img1.shape != img2.shape:
        img1 = resize(img1, img2.shape[:2][::-1], crop=False)

    MSE = mean_squared_error(img1, img2)
    if ycbcr:
        img1 = rgb2ycbcr(img1, True)
        img2 = rgb2ycbcr(img2, True)

    PSNR = peak_signal_noise_ratio(img1, img2, data_range=1)
    SSIM = structural_similarity(img1, img2, win_size=11, multichannel=False if ycbcr else True, data_range=1)
    
    return MSE, PSNR, SSIM

def compute_metrics(path1, path2, ycbcr=True):
    files1 = loadfiles(path1)
    files2 = loadfiles(path2)

    mse, psnr, ssim = [], [], []
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_metrics_for_pair, files1, files2, [path1]*len(files1), [path2]*len(files2),[ycbcr]*len(files2)), total=len(files1)))

    for result in results:
        MSE, PSNR, SSIM = result
        mse.append(MSE)
        psnr.append(PSNR)
        ssim.append(SSIM)

    mean_mse, mean_psnr, mean_ssim = np.mean(mse), np.mean(psnr), np.mean(ssim)
    print(mean_mse, mean_psnr, mean_ssim)
    return mean_mse, mean_psnr, mean_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('--path1', type=str,default= "") 
    parser.add_argument('--path2', type=str,default= "")
    args = parser.parse_args()

    path1 = args.path1
    path2 = args.path2

    compute_metrics(path1, path2, True)