#!/bin/bash
gpu_info=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)
echo "the gpu info is: $gpu_info"
# Default variable value
variable="aoyu"

# Check if an argument is provided
if [ "$1" != "" ]; then
    variable="$1"
fi

echo "The scenario of application is: $variable"

if [ "$variable" != "" ]; then
    case "$variable" in
        "aoyu")
            a100_dataset_list='/home/ec2-user/SageMaker/data/dataset/vto/shenin/labels/6_model_label_experiment' \
            a100_dataset_dir='/home/ec2-user/SageMaker/data/dataset/vto/shenin' \
            a10_dataset_list='/home/ubuntu/dataset/aigc-app-vto/shenin/labels/6_model_label_experiment' \
            a10_dataset_dir='/home/ubuntu/dataset/aigc-app-vto/shenin' \
            ;;
        "xiaoyu")
            a100_dataset_list='/home/ubuntu/VTO/dataset/shenin/test_pairs_shein_341.txt' \
            a100_dataset_dir='/home/ubuntu/VTO/dataset/shenin/' \
            a10_dataset_list='/home/ubuntu/VTO/dataset/shenin/test_pairs_shein_341.txt' \
            a10_dataset_dir='/home/ubuntu/VTO/dataset/shenin/' \
            ;;
        *)
            echo "Invalid option provided."
            exit 1
    esac
else
    echo "Please provide an option as a command-line argument."
    echo "Usage: $0 [opt1|opt2|opt3]"
    exit 1
fi

export USER="$variable"


/home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/anaconda3/envs/vto/bin/accelerate launch --num_processes=1 /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-atl/reference/IDM-VTON/inference.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --pretrained_model_name_or_path "/home/ubuntu/dataset/hf_cache/hub/models--yisol--IDM-VTON/snapshots/585a32e74aee241cbc0d0cc3ab21392ca58c916a" \
    --output_dir "result" \
    --unpaired \
    --data_dir "/home/ubuntu/dataset/aigc-app-vto/shenin" \
    --seed 42 \
    --test_batch_size 1 \
    --guidance_scale 2.0

# /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/anaconda3/envs/vto/bin/python /home/ubuntu/pytorch_gpu_base_ubuntu_uw2_workplace/aws-gcr-csdc-atl/aigc-vto-models/aigc-vto-models-atl/reference/IDM-VTON/inference_benchmark.py --pretrained_model_name_or_path "yisol/IDM-VTON" \
#     --width 768 \
#     --height 1024 \
#     --num_inference_steps 30   \
#     --enable_xformers_memory_efficient_attention \
#     --unpaired  \
#     --output_dir "unpair_shein_0" \
#     --data_dir "/home/ubuntu/dataset/aigc-app-vto/shenin/shein_data" \
#     --pair_txt_path /home/ubuntu/dataset/aigc-app-vto/shenin/test_pairs_shein.txt
