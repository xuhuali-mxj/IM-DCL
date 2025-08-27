## IM-DCL
The code of paper "Enhancing Information Maximization with Distance-Aware Contrastive Learning for Source-Free Cross-Domain Few-Shot Learning"

# 1. Setup
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


# 2. Code clone
git clone https://github.com/xuhuali-mxj/IM-DCL.git

cd IM-DCL

# 3. Dataset
For the 4 datasets CropDiseases, EuroSAT, ISIC, and ChestX, we refer to the [BS-CDFSL](https://github.com/IBM/cdfsl-benchmark) repo.

# 4. Run IM-DCL

## Based on ResNet

Our method aims at improving the performance of pretrained source model on the target FSL task. We introduce the information maximization, and propose a distance-aware contrastive learning, helping the pretrained source model to learn the decision boundary.

Please set your data address in [configs.py](configs.py).

We also provide the pretrained source model in mini_models/checkpoints/ResNet10_ce_aug/, We use ResNet10_ce_1200.tar to evaluate our IM-DCL.

We start from run.sh. Taking 5-way 1-shot as an example, the code runing process can be done as,

```
python ./adapt_da.py --model ResNet10 --train_aug --use_saved --dtarget CropDisease --n_shot 1
```

## Based on ViT
Down load the pretrained ViT from [here](https://github.com/google-research/vision_transformer). The models can be downloaded with e.g.:
```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
Run the following code:

```
python ./adapt_da_vit.py --model vit --train_aug --use_saved --dtarget CropDisease --n_shot 1
```

# 5. Acknowledge
Our code is built upon the implementation of [FTEM_BSR_CDFSL](https://github.com/liubingyuu/FTEM_BSR_CDFSL) and [SHOT](https://github.com/tim-learn/SHOT). Thanks for their work.
