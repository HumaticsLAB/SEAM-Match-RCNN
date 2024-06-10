<!--[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-networks-for-trajectory/trajectory-prediction-on-ethucy)](https://paperswithcode.com/sota/trajectory-prediction-on-ethucy?p=transformer-networks-for-trajectory)-->

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/movingfashion-a-benchmark-for-the-video-to/video-to-shop-on-movingfashion)](https://paperswithcode.com/sota/video-to-shop-on-movingfashion?p=movingfashion-a-benchmark-for-the-video-to)


# SEAM Match-RCNN
Official code of [**MovingFashion: a Benchmark for the Video-to-Shop Challenge**](https://arxiv.org/abs/2110.02627) paper

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]


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

conda create --name seam -y python=3
conda activate seam

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
## Dataset

SEAM Match-RCNN has been trained and test on MovingFashion and DeepFashion2 datasets.
Follow the instruction to download and extract the datasets.

We suggest to download the datasets inside the folder **data**.

### MovingFashion

MovingFashion dataset is available for academic purposes [here](https://bit.ly/4bTZGeS). 
<!--There's no need of pre-processing steps. The dataset is ready for use. -->

### Deepfashion2
DeepFashion2 dataset is available [here](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing). You need fill in the [form](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform?usp=sf_link) to get password for unzipping files.


Once the dataset will be extracted, use the reserved DeepFtoCoco.py script to convert the annotations in COCO format, specifying dataset path.
```bash
python DeepFtoCoco.py --path <dataset_root>
```



## Training
We provide the scripts to train both Match-RCNN and SEAM Match-RCNN. Check the scripts for all the possible parameters.

### Single GPU
```bash
#training of Match-RCNN
python train_matchrcnn.py --root_train <path_of_images_folder> --train_annots <json_path> --save_path <save_path> 

#training on movingfashion
python train_movingfashion.py --root <path_of_dataset_root> --train_annots <json_path> --test_annots <json_path> --pretrained_path <path_of_matchrcnn_model>


#training on multi-deepfashion2
python train_multiDF2.py --root <path_of_dataset_root> --train_annots <json_path> --test_annots <json_path> --pretrained_path <path_of_matchrcnn_model>
```


### Multi GPU
We use internally ```torch.distributed.launch``` in order to launch multi-gpu training. This utility function from PyTorch spawns as many Python processes as the number of GPUs we want to use, and each Python process will only use a single GPU.

```bash
#training of Match-RCNN
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> train_matchrcnn.py --root_train <path_of_images_folder> --train_annots <json_path> --save_path <save_path>

#training on movingfashion
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> train_movingfashion.py --root <path_of_dataset_root> --train_annots <json_path> --test_annots <json_path> --pretrained_path <path_of_matchrcnn_model> 

#training on multi-deepfashion2
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> train_multiDF2.py --root <path_of_dataset_root> --train_annots <json_path> --test_annots <json_path> --pretrained_path <path_of_matchrcnn_model> 
```


### Pre-Trained models
It is possibile to start training using the MatchRCNN pre-trained model.

**[MatchRCNN]** Pre-trained model on Deepfashion2 is available to download [here](https://bit.ly/3m3y6C4). This model can be used to start the training at the second phase (training directly SEAM Match-RCNN).

<!--**[SEAM Match-RCNN]** Pre-trained model on MovingFashion is available to download [here](http://bit.ly/...).  -->

<!-- **[SEAM Match-RCNN]** Pre-trained model on MultiDeepfashion2 is available to download [here](http://bit.ly/3j8Vc9W).-->

We suggest to download the model inside the folder **ckpt**. 

## Evaluation
To evaluate the models of SEAM Match-RCNN please use the following scripts.

```bash
#evaluation on movingfashion
python evaluate_movingfashion.py --root_test <path_of_dataset_root> --test_annots <json_path> --ckpt_path <checkpoint_path>


#evaluation on multi-deepfashion2
python evaluate_multiDF2.py --root_test <path_of_dataset_root> --test_annots <json_path> --ckpt_path <checkpoint_path>
```

## Citation
```
@misc{godi2021movingfashion,
      title={MovingFashion: a Benchmark for the Video-to-Shop Challenge}, 
      author={Marco Godi and Christian Joppi and Geri Skenderi and Marco Cristani},
      year={2021},
      eprint={2110.02627},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


