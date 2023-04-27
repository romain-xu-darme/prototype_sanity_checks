# Sanity checks for patch visualisation in prototype-based image classification
This repository contains the code developed for the experiments presented in the paper "Sanity checks for patch visualisation in prototype-based image classification", accepted at [XAI4CV](https://xai4cv.github.io/workshop_cvpr23) (2nd Explainable AI for Computer Vision Workshop at CVPR'23), including our modifications to the original code of:

* [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) by Chen *et al.*
* [ProtoTree](https://github.com/M-Nauta/ProtoTree) by Nauta *et al.*
* [PRP](https://github.com/SrishtiGautam/PRP) by Gautam *et al.*


## Dependencies
All dependencies are regrouped in the file requirements.txt. Simply run:
```bash
python -m pip install -r requirements.txt
```

## Setup
1. Download and preprocess the CUB200 and Stanford cars datasets, annotations and segmentation masks:
```bash
python prototree/preprocess_data/download_birds.py
python prototree/preprocess_data/cub.py
python prototree/preprocess_data/download_cars.py
python prototree/preprocess_data/cars.py
```
2. Download a ResNet50 model pretrained on the INaturalist dataset:
```bash
python prototree/features/get_state_dict.py 
```

## Experiments on ProtoTree
### On CUB200
1. Train a ProtoTree as follows
```bash
cd prototree
python main_tree.py \
	--num_features 256 --depth 9 --net resnet50_inat --init_mode pretrained --dataset CUB-200-2011 \
	--epochs 100 --lr 0.001 --lr_block 0.001 --lr_net 1e-5 --device cuda:0 \
	--freeze_epochs 10 --milestones 60,70,80,90,100 --batch_size 64 --random_seed 42 \
	--root_dir runs/prototree_cub  \
	--proj_dir proj_corners_sm --upsample_mode smoothgrads --upsample_threshold 0.3 --projection_mode corners
```
At the end of the training, this command will project prototypes using the augmented CUB dataset 
(images cropped to the four corners + center), and visualise prototypes using Smoothgrads.

Supported projection modes:
* raw: Use the raw training dataset
* corners: Training dataset augmented using 4 corners + center crop (CUB only)
* cropped: Training dataset cropped to object bounding box (CUB only)

Upsample mode is either:
* vanilla: Original upsampling with cubic interpolation
* smoothgrads: Use Smoothgrads x input
* prp: Use Prototype Relevance Propagation (PRP)

2. To perform inference and generate explanations on a test image, use:
```bash
python main_explain_local.py \
	--root_dir runs/prototree_cub/ --tree runs/prototree_cub/proj_raw_prp/model/ \
	--proj_dir proj_raw_prp/ --dataset CUB-200-2011 --device cuda:0 \
	--upsample_mode vanilla  \
	--sample_dir ../data/CUB_200_2011/dataset/test_full/054.Blue_Grosbeak/Blue_Grosbeak_0078_36655.jpg  \
	--results_dir explanations_vanilla
```

3. Generating fidelity and relevance statistics on prototypes.
```bash
python get_prototype_stats.py \
	--tree_dir runs/prototree_cub/checkpoints/latest/ \
	--base_arch resnet50  \
	--dataset CUB-200-2011 --use-segmentation \
	--output prototree_birds_r50_proto_stats.csv \
	--target_areas 0.001 0.02 0.001 --random_seed 0 \
	--projection_mode corners   \
	--proj_dir runs/prototree_cub/proj_corners
```

4. Generating fidelity and relevance statistics on test images.
```bash
python get_inference_stats.py \
	--tree_dir runs/prototree_cub/proj_corners_sm/model/ \
	--base_arch resnet50 \
	--dataset CUB-200-2011 --use-segmentation \
	--device cuda:0 \ 
	--output runs/prototree_cub/proj_corners/prototree_birds_r50_inference_stats.csv \
	--target_areas 0.001 0.02 0.001 --random_seed 0
```

### On Stanford Cars
1. Train a ProtoTree as follows
```bash
cd prototree
python main_tree.py \
	--num_features 128 --depth 9 --net resnet50 --init_mode pretrained \
	--dataset CARS \
	--epochs 500 --lr 0.001 --lr_block 0.001 --lr_net 2e-4 --device cuda:0 \
	--freeze_epochs 30 --milestones 250,350,400,425,450,475,500 \
	--batch_size 64 --random_seed 42 \
	--root_dir runs/prototree_cub  \
	--proj_dir proj_prp --upsample_mode prp \ 
	--upsample_threshold 0.3
```

2. To generate explanations and statistics, simply replace the CUB-200-2011 dataset with CARS.
Note that `--use-segmentation` is not available for Stanford Cars.

### Restart a training sequence from a checkpoint
There are two ways to restart a training sequence.
1. Relaunch the `main_tree.py` with exactly the same options. If the `--root_dir` directory exists, the training process 
will automatically restart from the checkpoint located in `<root_dir>/checkpoints/latest`.
2. Specify explicitely a path using the `--tree_dir` option pointing to the checkpoint directory.

### Perform different projection methods on a pretrained ProtoTree
It is possible to test different projection methods with different projection datasets without retraining the entire 
ProtoTree. Ex.
```bash
finalize_tree.py --tree_dir ./runs/prototree/checkpoints/latest/ \
	--root_dir runs/prototree \
	--dataset CUB-200-2011 \
	--device cuda:0 \
	--proj_dir proj_raw_vanilla \
	--upsample_threshold 0.98 --upsample_mode vanilla --projection_mode raw 
```

## Experiments on ProtoPNet

