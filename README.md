<!--[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-networks-for-trajectory/trajectory-prediction-on-ethucy)](https://paperswithcode.com/sota/trajectory-prediction-on-ethucy?p=transformer-networks-for-trajectory)-->
![PyTorch badge](https://img.shields.io/static/v1?label=pytorch&message=1.5.1&color=%3CCOLOR%3E)
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
mkdir data
mkdir ckpt

unset INSTALL_DIR
```
## Dataset and Pre-Trained models

SEAM Match-RCNN has been trained and test on MovingFashion and DeepFashion2 datasets.
Follow the instruction to download and extract the datasets.

We suggest to download the datasets inside the folder **data**.

### MovingFashion


### Deepfashion2
DeepFashion2 dataset is available [here](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing). You need fill in the [form](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform?usp=sf_link) to get password for unzipping files.


Once the dataset will be extracted, use the reserved DeepFtoCoco.py script to convert the annotations in COCO format, specifying dataset path.
```bash
python DeepFtoCoco.pt --path <dataset_root>
```

## Training

Qua va prima addestrato Match, poiSEAM
### Single GPU
python train_movingfashion

### Multi GPU

## Evaluation

## Citation
