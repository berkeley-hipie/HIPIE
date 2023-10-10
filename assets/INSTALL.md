# Install
## Requirements
We test the codes in the following environments, other versions may also be compatible but Pytorch vision should be >= 1.7

- CUDA 11.3
- Python 3.7
- Pytorch 1.12.1
- Torchvison 0.13.1

## Example conda envirnment
```
conda create --name hipie python=3.7 -y
conda activate hipie
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia
conda install -c anaconda jupyter
```

## Install dependencies for HIPIE

```
pip3 install -e . --user
pip3 install --user git+https://github.com/XD7479/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
pip3 install --user git+https://github.com/lvis-dataset/lvis-api.git
pip3 install --user git+https://github.com/cocodataset/panopticapi.git
pip3 install --user git+https://github.com/facebookresearch/segment-anything.git

# Compile Deformable Attention
cd projects/HIPIE/hipie/models/deformable_detr/ops
bash make.sh
cd ../../maskdino/pixel_decoder/ops
bash make.sh
cd ../../../../../../..
```

## Get Pretrained Weights 
*(BERT checkpoint is necessary in model initialization)*
Language Model (BERT-base)
```
mkdir -p projects/HIPIE/bert-base-uncased
cd projects/HIPIE/bert-base-uncased
wget -c https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget -c https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget -c https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
cd ../../..
```
