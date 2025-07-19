# BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting

### [[Project]]() [[Paper]](http://arxiv.org/abs/2506.22099) 

> [**BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting**](http://arxiv.org/abs/2506.22099)  
> [Zipei Ma](https://xiao10ma.github.io/)<sup>⚖</sup>, [Junzhe Jiang](https://selfspin.github.io/)<sup>⚖</sup>, [Yurui Chen](https://github.com/fumore), [Li Zhang](https://lzrobots.github.io)<sup>✉</sup>  
> **Shanghai Innovation Institute; School of Data Science, Fudan University**<br>
> **ICCV 2025**

**Official implementation of "BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting".** 

## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## 🎞️ Demo

**BézierGS.mp4**

[![BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, BézierGS.mp4 - YouTube](https://res.cloudinary.com/marcomontalbano/image/upload/v1751600146/video_to_markdown/images/youtube--lSMn9V2rBLc-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=lSMn9V2rBLc "BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, BézierGS.mp4 - YouTube")

**pedestrian.mp4**

[![BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, pedestrian.mp4 - YouTube](https://res.cloudinary.com/marcomontalbano/image/upload/v1751600597/video_to_markdown/images/youtube--sMb0xTdMumg-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=sMb0xTdMumg "BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, pedestrian.mp4 - YouTube")

## 🚀 Get started
### Environment
```bash
# Clone the repo.
git clone https://github.com/fudan-zvg/BezierGS
cd BezierGS

# Make a conda environment.
conda create --name bezier python=3.10
conda activate bezier

# Install requirements.
pip install -r requirements.txt

# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# a modified gaussian splatting (for feature rendering)
git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install nvdiffrast (for Envlight)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

### Data preparation

Create a directory for the data: `mkdir dataset`.

<details> <summary>Prepare Waymo Open Dataset.</summary>

We provide the split file following [EmerNeRF](https://github.com/NVlabs/EmerNeRF). You can refer to this [document](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md) for download details.

#### Preprocess the data

Preprocess the example scenes

```bash
python script/waymo/waymo_converter.py --root_dir TRAINING_SET_DIR --save_dir SAVE_DIR --split_file script/waymo/waymo_splits/demo.txt --segment_file script/waymo/waymo_splits/segment_list_train.txt
```

Generating LiDAR depth


```bash
python script/waymo/generate_lidar_depth.py --datadir DATA_DIR
```

Generating sky mask

Install GroundingDINO following this [repo](https://github.com/IDEA-Research/GroundingDINO) and download SAM checkpoint from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

```bash
python script/waymo/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CHECKPOINT
```

Generating intance segmentation
```bash
git clone https://github.com/xiao10ma/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
```
follow the instruction in the repo to install the dependencies.

Run the following command to generate the instance segmentation.
```bash
bash waymo_run.sh
```
</details>

### Training

```
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/waymo/017.yaml \
source_path=dataset/017 \
model_path=eval_output/waymo_nvs/017
```

After training, evaluation results can be found in `{EXPERIMENT_DIR}/eval_output` directory.

### Evaluating

You can also use the following command to evaluate.

```
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--config configs/waymo/017.yaml \
source_path=dataset/017 \
model_path=eval_output/waymo_nvs/017 \
checkpoint=eval_output/waymo_nvs/017/chkpnt30000.pth
```

## 📜 BibTeX

``` bibtex
@inproceedings{Ma2025BezierGS,
  title={BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting},
  author={Ma, Zipei and Jiang, Junzhe and Chen, Yurui and Zhang, Li},
  booktitle={ICCV},
  year={2025},
}
```
