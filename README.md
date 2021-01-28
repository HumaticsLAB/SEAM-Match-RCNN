# SEAM Match-RCNN
Official code of SEAM Match-RCNN: SElf Attention Multi-frame Match-RCNN for Street2Shop Matching using Video Sequences


## Installation

### Requirements:
- Pytorch 1.5.1 or more recent, with cudatoolkit (10.2)
- torchvision
- tensorboard
- cocoapi
- OpenCV Python
- tqdm
- cython
- CUDA >= 10

### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name seam_matchrcnn -y python=3
conda activate seam_matchrcnn

pip install cython tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

conda install tensorboard

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# download SEAM
cd $INSTALL_DIR
git clone https://github.com/VIPS4/SEAM-Match-RCNN.git
cd SEAM-Match-RCNN

unset INSTALL_DIR
```
## Dataset and Pre-Trained models

SEAM Match-RCNN has been trained and test on MovingFashion and DeepFashion2 datasets.
Follow the instruction to download and extract the datasets.

### MovingFashion


### Deepfashion


## Training

Qua va prima addestrato Match, poiSEAM
### Single GPU
python train_movingfashion

### Multi GPU

## Evaluation

## Citation
