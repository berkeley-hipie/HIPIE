# Data Preparation

## Pretrained Weights
Language Model (BERT-base)
```
mkdir -p projects/HIPIE/bert-base-uncased
cd projects/HIPIE/bert-base-uncased
wget -c https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget -c https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget -c https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
cd ../../..
```

Visual Backbones
```
mkdir -p weights
cd weights
# R50
wget -c https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl
# ConvNeXt-Large
wget -c https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth
# Convert ConvNeXt-Large
cd ..
python3 conversion/convert_convnext.py --source_model weights/convnext_large_22k_1k_384.pth --output_model weights/convnext_large_22k_1k_384_new.pth
python3 projects/HIPIE/convert_pth2pkl.py weights/convnext_large_22k_1k_384_new.pth weights/convnext_large_22k_1k_384_new.pkl
# ViT-Huge
wget -c https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MAE/mae_pretrain_vit_huge_p14to16.pth
# Swin
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
python ../tools/convert-pretrained-swin-model-to-d2.py swin_large_patch4_window12_384_22k.pth swin_large_patch4_window12_384_22k.pkl

```



Other pretrained models can be found in [MODEL_ZOO.md](MODEL_ZOO.md)

## Data
For users who are only interested in part of tasks, there is no need of downloading all datasets. The following lines list the datasets needed for different tasks. Datasets in the brackets are only used during the inference.

- **Pretrain**: Objects365
- **Object detection & instance segmentation**: COCO2017
- **REC & RES**: RefCOCO, RefCOCOg, RefCOCO+
- **SOT**: COCO, LaSOT, GOT-10K, TrackingNet, (LaSOT-ext, TNL-2k)
- **VOS**: Youtube-VOS 2018, COCO, (DAVIS)
- **MOT & MOTS**: COCO, BDD100K
- **VIS**: COCO, Youtube-VIS 2019, OVIS
- **R-VOS**: RefCOCO, RefCOCOg, RefCOCO+, Ref-Youtube-VOS, (Ref-DAVIS)



### Pretraining
Pretraining on Objects365 requires many training resources. For HIPIE-50, Objects365 pretraining needs 3~4 days on 32 A100 GPUs. Thus we suggest users directly loading provided weights instead of re-running this step. If users still want to use this dataset, we provide a script for automatically downloading images of Objects365 V2.
```
python3 conversion/download_obj365_v2.py
```
Following DINO, we select the first 5,000 out of 80,000 validation images as our
validation set and add the others to training. We put the processed json files on [Google Drive](), which can be directly downloaded.
We expect that the data is organized as below.
```
${HIPIE_ROOT}
    -- datasets
        -- Objects365
            -- annotations
                -- zhiyuan_objv2_train_new.json
                -- zhiyuan_objv2_val_new.json
            -- images
```

### Object Detection & Instance Segmentation
Please download [COCO](https://cocodataset.org/#home) from the offical website. We use [train2017.zip](http://images.cocodataset.org/zips/train2017.zip), [val2017.zip](http://images.cocodataset.org/zips/val2017.zip), [test2017.zip](http://images.cocodataset.org/zips/test2017.zip) & [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [image_info_test2017.zip](http://images.cocodataset.org/annotations/image_info_test2017.zip). We expect that the data is organized as below.
```
${HIPIE_ROOT}
    -- datasets
        -- coco
            -- annotations
            -- train2017
            -- val2017
            -- test2017
```

### REC & RES
Please download processed json files by [SeqTR](https://github.com/sean-zhuh/SeqTR) from [Google Drive](https://drive.google.com/drive/folders/1IXnSieVr5CHF2pVJpj0DlwC6R3SbfolU). We need three folders: refcoco-unc, refcocog-umd, and refcocoplus-unc. These folders should be organized as below.
```
${HIPIE_ROOT}
    -- datasets
        -- annotations
            -- refcoco-unc
            -- refcocog-umd
            -- refcocoplus-unc
```
Then run ```python3 conversion/convert_mix_ref.py``` to convert the original jsons to COCO format and merge them into one dataset ```refcoco-mixed```. Besides, please download images of [COCO2014 train](http://images.cocodataset.org/zips/train2014.zip) and put ```train2014``` folder under ```datasets/coco```. 


### Panoptic Segmentation and Open Vocabulary Senmantic Segmentation 

Please follow [ODISE setup](https://github.com/NVlabs/ODISE/blob/main/datasets/README.md) for COCO_Panoptic, ADE20k,ADE-Full,Pascal VOC, PASCAL-Context
```
The data should be organized as below.
${HIPIE_ROOT}
    -- datasets
        -- coco
            -- annotations
            -- panoptic_semseg_train2017
            -- panoptic_semseg_val2017
            -- panoptic_semseg_val2017_100
            -- panoptic_train2017
            -- panoptic_val2017
            -- panoptic_val2017_100
            ... other data 
        -- ade
            -- ADE20K_2021_17_01
                -- annotations_detectron2
                -- images
                -- images_detectron2
                ... other data 
            -- ADEChallengeData2016
                -- ade20k_panoptic_train
                -- ade20k_panoptic_val
                -- annotations
                -- annotations_detectron2
                -- annotations_instance
                -- images
                -- ade20k_instance_train.json
                -- ade20k_instance_val.json
                -- ade20k_panoptic_train.json
                -- ade20k_panoptic_val.json
                -- objectInfo150.txt
                -- sceneCategories.txt
        -- pascal_ctx_d2
            -- annotations_ctx59
            -- annotations_ctx459
            -- images
        -- pascal_voc_d2
            -- annotations_pascal21
            -- images

```

### Paco Dataset
Please follow [PACO](https://github.com/facebookresearch/paco)
```
The data should be organized as below.
${HIPIE_ROOT}
    -- datasets
        -- ego4d
            -- v1
                -- paco_frames
            -- ego4d.json
        -- coco
            -- train2017
            -- val2017
        -- paco
            paco_lvis_v1_train.json
            paco_ego4d_v1_train.json
            ... <other annotations in paco dataset>
```

### Odinw and SeginW
Please download dataset and put them in following file structure

```
The data should be organized as below.
${HIPIE_ROOT}
    -- datasets
        -- seginw
            -- Airplan-Parts
                -- train
                -- train_10shot
                -- valid
            -- Bottles
                -- train
                -- train_10shot
                -- valid
            <other datasets>
        -- odinw
            -- AerialMaritimeDrone
                -- large
                -- tiled
            -- AmericanSignLanguageLetters
                -- test
                -- train
                -- valid
            <other datasets>

```

### Pascal Parts
Please download dataset [Here](https://drive.google.com/file/d/1Y1069ShPL9rH8PJcXmApBXBnChOr_s-1/view?usp=drive_link)

