#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROJECT/anaconda3/lib
module load cuda cudnn nvhpc
export CUDA_HOME=/jet/packages/cuda/v11.7.1
nvidia-smi
source ~/.bashrc
conda activate csc791
cd /jet/home/bpark1/csc791-025/final

# These don't work
# FMEN
# VDSR

# These work
# IMDN
# RDN
# RFDN
# SuperResolutionByteDance
# SuperResolutionTwitter
# WDSR

### Train models
MODEL="SuperResolutionTwitter"

# for i in 2 3 4 5
# do
#     python3 final.py --mode training --upscale_factor $i --model_name $MODEL
#     python3 final.py --mode inference --model_path /ocean/projects/cis220070p/bpark1/models/${MODEL}/original/${i}/model_epoch_100.pth --upscale_factor $i
# done


### Compile baseline ONNX models for XGen
for SR_MODEL in "IMDN" "RDN" "RFDN" "SuperResolutionByteDance" "SuperResolutionTwitter" "WDSR"
do 
    # Factor in TVM later (save the ansor version)
    # python3 final.py --mode tvm --upscale_factor 4 --model_path $PROJECT/models/${SR_MODEL}/original/4/model_epoch_100.pth
    # python3 final.py --mode onnx --upscale_factor 4 --model_path $PROJECT/models/${SR_MODEL}/original/4/model_epoch_100.pth
    python3 final.py --mode coreml --upscale_factor 4 --model_path $PROJECT/models/${SR_MODEL}/original/4/model_epoch_100.pth
    # python3 final.py --mode tensorrt --upscale_factor 4 --model_path $PROJECT/models/${SR_MODEL}/original/4/model_epoch_100.pth
done
# $PROJECT/models/SuperResolutionTwitter/LevelPruner/4/0.6/model_epoch_100.pth


### Prune Models
# UPSCALE_FACTOR=4
# MODEL_PATH=/ocean/projects/cis220070p/bpark1/models/${MODEL}/original/${UPSCALE_FACTOR}/model_epoch_100.pth
# for PRUNER in "LevelPruner" "L1NormPruner" "L2NormPruner" "FPGMPruner" "ActivationAPoZRankPruner" "TaylorFOWeightPruner" "ADMMPruner"
# do 
#     for SPARSITY in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#     do
#         PRUNED_MODEL_PATH=/ocean/projects/cis220070p/bpark1/models/${MODEL}/${PRUNER}/${UPSCALE_FACTOR}/${SPARSITY}/model_epoch_10.pth
#         python3 final.py --mode prune --upscale_factor $UPSCALE_FACTOR --model_path $MODEL_PATH --sparsity $SPARSITY --pruner $PRUNER
#         python3 final.py --mode inference --model_path $PRUNED_MODEL_PATH --upscale_factor $UPSCALE_FACTOR --pruner $PRUNER --sparsity $SPARSITY
#     done
# done

# # Process all models
# for model in "IMDN" "RDN" "RFDN" "SuperResolutionByteDance" "SuperResolutionTwitter" "WDSR"
# do
#     for i in 2 3 4 5
#     do
#         python3 final.py --mode inference --upscale_factor $i --model_path /ocean/projects/cis220070p/bpark1/models/${model}/original/${i}/model_epoch_1000.pth
#     done
# done



# python ./prepare_dataset.py --images_dir ../data/Urban100/original --output_dir ../data/Urban100/VDSR/train --image_size 44 --step 44 --scale 4 --num_workers 10

# python ./split_train_valid_dataset.py --train_images_dir ../data/Urban100/VDSR/train  --valid_images_dir ../data/Urban100/VDSR/valid --valid_samples_ratio 0.1

# python3 final.py --mode demo --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionTwitter/original/4/model_epoch_100.pth
# ffmpeg -i video/macross.mp4 -vf fps=30 frames/macross/out%6d.png
# ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4
# /ocean/projects/cis220070p/bpark1/demo/frames/macross



# python3 final.py --mode training --upscale_factor 4 --model_name FMEN
# python3 final.py --mode training --upscale_factor 4 --model_name VDSR

# python3 final.py --mode training --upscale_factor 4 --model_name IMDN
# python3 final.py --mode training --upscale_factor 4 --model_name RDN
# python3 final.py --mode training --upscale_factor 4 --model_name RFDN
# python3 final.py --mode training --upscale_factor 4 --model_name SuperResolutionByteDance
# python3 final.py --mode training --upscale_factor 4 --model_name SuperResolutionTwitter
# python3 final.py --mode training --upscale_factor 4 --model_name WDSR


# python3 final.py --mode inference --upscale_factor 4 --model_path /ocean/projects/cis220070p/bpark1/models/IMDN/original/4/model_epoch_10.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path /ocean/projects/cis220070p/bpark1/models/RDN/original/4/model_epoch_10.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path /ocean/projects/cis220070p/bpark1/models/RFDN/original/4/model_epoch_10.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path /ocean/projects/cis220070p/bpark1/models/SuperResolutionByteDance/original/4/model_epoch_10.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path /ocean/projects/cis220070p/bpark1/models/SuperResolutionTwitter/original/4/model_epoch_1000.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path /ocean/projects/cis220070p/bpark1/models/WDSR/original/4/model_epoch_10.pth




# # Inference
# python3 final.py --mode inference --upscale_factor 4 --model_path $PROJECT/models/IMDN/original/4/model_epoch_1000.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path $PROJECT/models/RDN/original/4/model_epoch_1000.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path $PROJECT/models/RFDN/original/4/model_epoch_1000.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionByteDance/original/4/model_epoch_1000.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionTwitter/original/4/model_epoch_1000.pth
# python3 final.py --mode inference --upscale_factor 4 --model_path $PROJECT/models/WDSR/original/4/model_epoch_1000.pth

# # ONNX
# python3 final.py --mode onnx --upscale_factor 4 --model_path $PROJECT/models/IMDN/original/4/model_epoch_1000.pth
# python3 final.py --mode onnx --upscale_factor 4 --model_path $PROJECT/models/RDN/original/4/model_epoch_1000.pth
# python3 final.py --mode onnx --upscale_factor 4 --model_path $PROJECT/models/RFDN/original/4/model_epoch_1000.pth
# python3 final.py --mode onnx --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionByteDance/original/4/model_epoch_1000.pth
# python3 final.py --mode onnx --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionTwitter/original/4/model_epoch_1000.pth
# python3 final.py --mode onnx --upscale_factor 4 --model_path $PROJECT/models/WDSR/original/4/model_epoch_1000.pth

# # TensorRT
# python3 final.py --mode tensorrt --upscale_factor 4 --model_path $PROJECT/models/IMDN/original/4/model_epoch_1000.pth
# python3 final.py --mode tensorrt --upscale_factor 4 --model_path $PROJECT/models/RDN/original/4/model_epoch_1000.pth
# python3 final.py --mode tensorrt --upscale_factor 4 --model_path $PROJECT/models/RFDN/original/4/model_epoch_1000.pth
# python3 final.py --mode tensorrt --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionByteDance/original/4/model_epoch_1000.pth
# python3 final.py --mode tensorrt --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionTwitter/original/4/model_epoch_1000.pth
# python3 final.py --mode tensorrt --upscale_factor 4 --model_path $PROJECT/models/WDSR/original/4/model_epoch_1000.pth

# TVM
# python3 final.py --mode tvm --upscale_factor 4 --model_path $PROJECT/models/IMDN/original/4/model_epoch_1000.pth
# python3 final.py --mode tvm --upscale_factor 4 --model_path $PROJECT/models/RDN/original/4/model_epoch_1000.pth
# python3 final.py --mode tvm --upscale_factor 4 --model_path $PROJECT/models/RFDN/original/4/model_epoch_1000.pth
# python3 final.py --mode tvm --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionByteDance/original/4/model_epoch_1000.pth
# python3 final.py --mode tvm --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionTwitter/original/4/model_epoch_1000.pth
# python3 final.py --mode tvm --upscale_factor 4 --model_path $PROJECT/models/WDSR/original/4/model_epoch_1000.pth



# python3 final.py --mode benchmark --upscale_factor 4 --model_path $PROJECT/models/SuperResolutionTwitter/original/4/model_epoch_1000.pth