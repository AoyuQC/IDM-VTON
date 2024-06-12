import torch

device_name = torch.cuda.get_device_name()
epoch_num = 0
if device_name == 'NVIDIA A10G':
    try:
        import debugpy

        debugpy.listen(5889)  # 5678 is port
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')
    except:
        print("non debug mode")
    
    base_path = f"/home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-atl/reference/aigc-vto-models-atl/vto-train/checkpoint-split-epoch{epoch_num}/"
    infer_base_path = "/home/ubuntu/dataset/hf_cache/hub/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a/"
elif device_name == 'NVIDIA A100-SXM4-40GB':
    # a100 instance
    base_path = f"/home/ec2-user/SageMaker/vto/OOTDiffusion-train/ootd_train_ds_checkpoints/checkpoint-split-epoch{epoch_num}"
    infer_base_path = "/home/ec2-user/SageMaker/vto/OOTDiffusion"
else:
    raise Exception("only for a10 and a100 instance")

from safetensors.torch import save_file
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(base_path)
net_name = set()
unet_vton_dict = dict()
unet_encoder_dict = dict()
unet_len = []
unet_encoder_len = []
for idx, key in enumerate(state_dict):
        name = key.split('.')[0]
        net_name.add(name)
        if name == 'unet_vton':
                new_key = key[10:]
                unet_vton_dict[new_key] = state_dict[key]
                unet_len.append(name)
        elif name == 'unet_encoder':
                new_key = key[13:]
                unet_encoder_dict[new_key] = state_dict[key]
                unet_encoder_len.append(name)

save_file(unet_vton_dict, "unet_vton.safetensors")
save_file(unet_encoder_dict, "unet_encoder.safetensors")

# make upper body folder and soft link
import os
import shutil
from pathlib import Path

# Create a new folder
unet_folder_name = f"unet_epoch{epoch_num}"
os.makedirs(os.path.join(infer_base_path,unet_folder_name), exist_ok=True)
unet_encoder_folder_name = f"unet_encoder_epoch{epoch_num}"
os.makedirs(os.path.join(infer_base_path,unet_encoder_folder_name), exist_ok=True)
# unet_garm_folder_name = f"checkpoints/ootd/ootd_hd/checkpoint-epoch{epoch_num}/unet_garm"
# os.makedirs(os.path.join(infer_base_path,unet_garm_folder_name), exist_ok=True)
# folder_name = f"unet_epoch{epoch_num}"
# unet_vton_folder_name = f"checkpoints/ootd/ootd_hd/checkpoint-epoch{epoch_num}/unet_vton"
# os.makedirs(os.path.join(infer_base_path,unet_vton_folder_name), exist_ok=True)

source_path = "unet/config.json"
shutil.copy(os.path.join(infer_base_path, source_path), os.path.join(infer_base_path, unet_folder_name,"config.json"))
source_path = "unet_encoder/config.json"
shutil.copy(os.path.join(infer_base_path, source_path), os.path.join(infer_base_path, unet_encoder_folder_name,"config.json"))

dst_path = os.path.join(infer_base_path, unet_folder_name, "diffusion_pytorch_model.safetensors")
shutil.move("unet_vton.safetensors", dst_path)
dst_path = os.path.join(infer_base_path, unet_encoder_folder_name, "diffusion_pytorch_model.safetensors")
shutil.move("unet_encoder.safetensors", dst_path)
