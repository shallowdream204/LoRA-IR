## Dataset Preparation
Please download the allweather dataset provided in [Histoformer](https://github.com/sunshangquan/Histoformer).

## Train

1. To train DG-Router with default settings, run
```shell
cd LoRA-IR/classification
python3 train_setting1.py --path_snow ... --path_rain ... --path_raindrop ...
```

2. To pretrain LoRA-IR with default settings, run
```shell
cd LoRA-IR
# Don't forget to modify the specific data&ckpt path in the yaml file
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 \
basicsr/train.py -opt Setting1/options/Setting1_Pre.yml --launcher pytorch
```

3. To fine-tune LoRA-IR with default settings, run
```shell
cd LoRA-IR
# Don't forget to modify the specific data&ckpt path in the yaml file
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 \
basicsr/train.py -opt Setting1/options/Setting1_Ft.yml --launcher pytorch
```

## Evaluation
1. Download the pre-trained [DG-Router](https://huggingface.co/shallowdream204/LoRA-IR/resolve/main/setting1_router.pth) and [IR](https://huggingface.co/shallowdream204/LoRA-IR/resolve/main/setting1_ir.pth) models.

2. For testing DR-Router, run
```shell
cd LoRA-IR/classification
# for cls label, snow=0, rain=1, raindrop=2
python3 test_setting1.py --ckpt /path/to/setting1_router.pth \
--path /path/to/lq/image/ --label 0/1/2
```

3. For testing IR model, run
```shell
cd LoRA-IR
python3 test_3input.py --input_dir /path/to/lq/image/ \
--result_dir /path/to/save/dir \
--weights /path/to/setting1_ir.pth \
--weights_clip /path/to/setting1_router.pth \
--yaml_file Setting1/options/Setting1_Ft.yml
```

#### To reproduce PSNR/SSIM results of Tab. 1, run
```shell
cd LoRA-IR/Setting1
python3 compute_psnr_ssim.py --path1 ... --path2 ...
```
