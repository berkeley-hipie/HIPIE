# HIPIE: Hierarchical Open-vocabulary Universal Image Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-open-vocabulary-universal-image-1/referring-expression-segmentation-on-refcoco)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco?p=hierarchical-open-vocabulary-universal-image-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-open-vocabulary-universal-image-1/referring-expression-segmentation-on-refcoco-3)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcoco-3?p=hierarchical-open-vocabulary-universal-image-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-open-vocabulary-universal-image-1/image-segmentation-on-pascal-panoptic-parts)](https://paperswithcode.com/sota/image-segmentation-on-pascal-panoptic-parts?p=hierarchical-open-vocabulary-universal-image-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-open-vocabulary-universal-image-1/zero-shot-segmentation-on-segmentation-in-the)](https://paperswithcode.com/sota/zero-shot-segmentation-on-segmentation-in-the?p=hierarchical-open-vocabulary-universal-image-1) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hierarchical-open-vocabulary-universal-image-1/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=hierarchical-open-vocabulary-universal-image-1) 


We present **HIPIE**, a novel HIerarchical, oPen-vocabulary and unIvErsal image segmentation and detection model that is capable of performing segmentation tasks at various levels of granularities (whole, part and subpart) and tasks, including semantic segmentation, instance segmentation, panoptic segmentation, referring segmentation, and part/subpart segmentation, all within a unified framework of language-guided segmentation. 

<p align="center"> <img src='assets/teaser.png' align="center" > </p>            

> [**Hierarchical Open-vocabulary Universal Image Segmentation**](http://people.eecs.berkeley.edu/~xdwang/projects/HIPIE/)            
> [Xudong Wang*](https://people.eecs.berkeley.edu/~xdwang/), [Shufan Li*](https://homepage.jackli.org/), [Konstantinos Kallidromitis*](https://tech-ai.panasonic.com/en/researcher_introduction/048/), Yusuke Kato, Kazuki Kozuka, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)            
> Berkeley AI Research, UC Berkeley; Panasonic AI Research    
> NeurIPS 2023

[[`project page`](http://people.eecs.berkeley.edu/~xdwang/projects/HIPIE/)] [[`arxiv`](https://arxiv.org/abs/2307.00764)] [[`paper`](https://arxiv.org/pdf/2307.00764.pdf)] [[`bibtex`](#citation)]

*Oct 5: We release more weights, and codes for training and evaluation*  

*Oct 15: We release additional Vit-H weights finetuned for part segmentation* 

## Installation
Please refer to [INSTALL.md](assets/INSTALL.md) for more details.  


## Demos
-  See  [Demo-Main](notebooks/Demo-Main.ipynb) for Panoptic, Part, Instance and Referring Segmentation
-  See  [Demo-SD](notebooks/Inpaint.ipynb) for Combining our model with Stable Diffusion
-  See  [Demo-SAM](notebooks/Demo-HIPIE+SAM.ipynb) for Combining our model with Segment Anything

<p align="center">
  <img src="https://github.com/frank-xwang/assets/blob/d45c1b4a44dcfda9a84f54c6fbb12dab9b578c74/HIPIE-Demos-Github.gif" width=100%>
</p>
HIPIE is also capable of labeling segmentation masks from SAM and can even identify additional masks that may have been overlooked by SAM.

Please check our [project page](http://people.eecs.berkeley.edu/~xdwang/projects/HIPIE/) for more demos!

## Model Zoo
We release the following [checkpoints](https://huggingface.co/KonstantinosKK/HIPIE) at the moment.

- ResNet-50 Pretrained with O365,COCO,RefCOCO,Pascal Panoptic Parts
- ViT-H Pretrained with O365,COCO,RefCOCO,PACO
- ViT-H Finetuned on COCO,RefCOCO,Pascal-Parts

## Training

The following code will train model on one node with 8 A100 GPUS
```
python3 launch.py --nn 1 --np 8 --uni 1 --config-file projects/HIPIE/configs/<config file>  MODEL.WEIGHTS <pretrained checkpoint>
```


## Evaluation
The following code will evaluate model on one node with 8 A100 GPUS
```
python3 launch.py --nn 1 --np 8 --uni 1 --config-file projects/HIPIE/configs/<config file> --eval-only  MODEL.WEIGHTS < checkpoint to load>
```

Alternatively, one can run

```
python3 launch.py --nn 1 --np 8 --uni 1 --config-file projects/HIPIE/configs/<config file> --eval-only --resume OUTPUT_DIR < folder with checkpoints >
```

with released weights, on should be able to reproduce following results

<table >
<thead >
  <tr>
    <th  > Data</th>
    <th  colspan='4'>COCO</th>
    <th  colspan='4'>ADE-150</th>
    <th >RefCOCO</th>
    <th >RefCOCO+</th>
    <th >RefCOCOg</th>
    <th >PAS-21</th>
    <th >CTX-59</th>
    <th >CTX-459</th>
    <th >ADE-874</th>
  </tr>
</thead>
<tbody>
  <tr>
  <td> </td>
    <td >AP_bbox</td>
    <td >AP_segm</td>
    <td >MIoU</td>
    <td >PQ</td>
    <td >AP_bbox</td>
    <td >AP_segm</td>
    <td >MIoU</td>
    <td >PQ</td>
    <td >oIoU</td>
    <td >oIoU</td>
    <td >oIoU</td>
    <td >MIoU</td>
    <td >MIoU</td>
    <td >MIoU</td>
    <td >MIoU</td>
  </tr>
  <tr>
   <td> O365, COCO, RefCOCO/+/g,PACO</td>
    <td >60.4</td>
    <td >51.1</td>
    <td >65.6</td>
    <td >57.0</td>
    <td >23.0</td>
    <td >19.1</td>
    <td >24.3</td>
    <td >21.0</td>
    <td >81.5</td>
    <td >71.5</td>
    <td >74.3</td>
    <td >81.1</td>
    <td >57.4</td>
    <td >14.4</td>
    <td >9.7</td>
  </tr>
  <tr>
   <td> O365*, COCO, RefCOCO/+/g</td>
    <td >61.3</td>
    <td >51.9</td>
    <td >66.8</td>
    <td >58.1</td>
    <td >18.4</td>
    <td >14.9</td>
    <td >28.4</td>
    <td >20.1</td>
    <td >82.8</td>
    <td >73.9</td>
    <td >75.7</td>
    <td >83.2</td>
    <td >58.1</td>
    <td >11.1</td>
    <td >10.8</td>
  </tr>
</tbody>
</table>

\* Used only in pretraing, but not in final training.

\** Note on high variance: We observe that evaluation metrics can have high variances, this is likely due to the noise of using CLIP MODEL. Specifically, changing the `MODEL.CLIP.ALPHA` and `MODEL.CLIP.BETA` which determines the importances of CLIP feature versus encoder feature can drastically change the results. It is possible to improve on individual benchmark by tuning these parameters. 

The finetuned part segmentation model should be able to produce the following result

<table >
<thead >
  <tr>
    <th > COCO</th>
    <th > RefCOCO</th>
    <th > Pascal-Parts</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td >PQ</td>
    <td >oIoU</td>
    <td >MIoU-PastS</td>
  </tr>
    <tr>
    <td >55.3</td>
    <td >78.1</td>
    <td >64.4</td>
  </tr>
  </tbody>
</table>


## License
The majority of HIPIE is licensed under the [MIT license](LICENSE). If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than CC-BY-NC, MIT, or CC0.


## How to get support from us?
If you have any general questions, feel free to email us at [Xudong Wang](mailto:xdwang@eecs.berkeley.edu), [Shufan Li](mailto:jacklishufan@berkeley.edu) and [Konstantinos Kallidromitis](mailto:kk984@cornell.edu). If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 


## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{wang2023hierarchical,
  title={Hierarchical Open-vocabulary Universal Image Segmentation},
  author={Wang, Xudong and Li, Shufan and Kallidromitis, Konstantinos and Kato, Yusuke and Kozuka, Kazuki and Darrell, Trevor},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
