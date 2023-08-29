## IM-DCL
The code of paper "Enhancing Information Maximization with Distance-Aware Contrastive Learning for Source-Free Cross-Domain Few-Shot Learning"

![Overview of IM-DCL](img-folder/1693295248(1).png)

## Setup
conda creat --name im-dcl python=3.9
conda activate im-dcl
conda install pytorch torchvision -c pytorch
conda install pandas
pip install numpy
pip install argparse
pip install math
pip install os
pip install sklearn
pip install scipy
pip install PIL
pip install abc

# Code clone
git clone https://github.com/xuhuali-mxj/IM-DCL.git
cd IM-DCL

# Dataset
For the 4 datasets CropDiseases, EuroSAT, ISIC, and ChestX, we refer to the [BS-CDFSL](https://github.com/IBM/cdfsl-benchmark) repo.
