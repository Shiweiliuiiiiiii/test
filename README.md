ICLR 2023 Submission: More ConvNets in the 2020s: Scaling up Kernels Beyond 51 × 51 using Sparsity


## Installation

The code is tested used CUDA 11.3.1, cudnn 8.2.0, PyTorch 1.10.0 with A100 GPUs.

### Dependency Setup
Create an new conda virtual environment
```
conda create -n slak python=3.8 -y
conda activate slak
```

Install [Pytorch](https://pytorch.org/)>=1.10.0. For example:
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Install required packages:
```
pip install timm tensorboardX six
```

To enable training SLaK, we follow [RepLKNet] and install the efficient large-kernel convolution with PyTorch provided by MegEngine:

1. ```cd cutlass/examples/19_large_depthwise_conv2d_torch_extension```
2. ```./setup.py install --user```. If you get errors, (1) check your ```CUDA_HOME```; (2) you might need to change the source code a bit to make tensors contiguous. 
3. A quick check: ```python depthwise_conv2d_implicit_gemm.py```
4. Add ```WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` into your ```PYTHONPATH``` so that you can ```from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM``` anywhere. Then you may use ```DepthWiseConv2dImplicitGEMM``` as a replacement of ```nn.Conv2d```.
5. ```export LARGE_KERNEL_CONV_IMPL=WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` so that RepLKNet will use the efficient implementation. Or you may simply modify the related code (```get_conv2d```) in ```SLaK.py```.

## Training code

We provide ImageNet-1K training, and ImageNet-1K fine-tuning commands here.

### ImageNet-1K SLaK-T on a single machine
```
python -m torch.distributed.launch --nproc_per_node=4 main.py  \
--Decom True --sparse --width_factor 1.3 -u 2000 --sparsity 0.4 --sparse_init snip  --prune_rate 0.5 --growth random \
--epochs 300 --model SLaK_tiny --drop_path 0.1 --batch_size 128 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 40 \
--kernel_size 51 49 47 13 5 --output_dir /path/to/save_results
```

- **To enable to train/evaluate SLaK models, make sure that you add `--sparse --Decom True --kernel_size 51 49 47 13 5 --sparse_init snip` in your script.** `--sparse`: enable sparse model; `--sparsity`: model sparsity; `--width_factor`: model width; `-u`: adaptation frequency; `--prune_rate`: adaptation rate, `--kernel_size`: [4 * (kernel size of each stage) + the size of the smaller kernel edge].
- You can add `--use_amp true` to train in PyTorch's Automatic Mixed Precision (AMP).
- Use `--resume /path_or_url/to/checkpoint.pth` to resume training from a previous checkpoint; use `--auto_resume true` to auto-resume from latest checkpoint in the specified output folder. To resume the training of sparse models, we need to set `--sparse_init resume` to get the masks.
- `--batch_size`: batch size per GPU; `--update_freq`: gradient accumulation steps.
- The effective batch size = `--nodes` * `--ngpus` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `4*8*128*1 = 4096`. You can adjust these four arguments together to keep the effective batch size at 4096 and avoid OOM issues, based on the model size, number of nodes and GPU memory.

### ImageNet-1K SLaK-S on a single machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py  \
--Decom True --sparse --width_factor 1.3 -u 100 --sparsity 0.4 --sparse_init snip  --prune_rate 0.3 --growth random \
--epochs 300 --model SLaK_small --drop_path 0.4 --batch_size 64 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 40 \
--kernel_size 51 49 47 13 5 --output_dir /path/to/save_results
```

### ImageNet-1K SLaK-B on a single machine
```
python -m torch.distributed.launch --nproc_per_node=16 main.py  \
--Decom True --sparse --width_factor 1.3 -u 100 --sparsity 0.4 --sparse_init snip  --prune_rate 0.3 --growth random \
--epochs 300 --model SLaK_base --drop_path 0.5 --batch_size 32 \
--lr 4e-3 --update_freq 8 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k --num_workers 40 \
--kernel_size 51 49 47 13 5 --output_dir /path/to/save_results
```

To run ConvNeXt, simple set the kernel size as --kernel_size 7 7 7 7 100. (Make sure that the last number is larger than the first four numbers)

## Evaluation
We give an example evaluation command for a SLaK_tiny on ImageNet-1K :

Single-GPU
```
python main.py --model SLaK_tiny --eval true \
--Decom True --kernel_size 51 49 47 13 5 --width_factor 1.3 \
--resume path/to/checkpoint \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```

Multi-GPUs
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model SLaK_tiny --eval true \
--Decom True --kernel_size 51 49 47 13 5 --width_factor 1.3 \
--resume path/to/checkpoint \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```
