import torch
import PIL
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, repeat

from detectron2.config import get_cfg
from detectron2.projects.hipie import add_hipie_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from matplotlib import pyplot as plt
from projects.HIPIE.predictor import VisualizationDemo
from argparse import Namespace
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.projects.hipie.data.coco_dataset_mapper_uni import get_openseg_labels,cat2ind_panoptics_coco,create_queries_and_maps 

@torch.no_grad()
def vote(masks,senmatic_map,out_size):
    '''
    masks: C X H X W or H X W
    segment_map: C X H X W or H X W
    out_size: H,W
    '''
    if len(senmatic_map.shape) == 2:
        senmatic_map = F.one_hot(senmatic_map)
        senmatic_map = rearrange(senmatic_map,'h w c -> c h w')
    
    if len(masks.shape) == 2:
        masks = F.one_hot(masks)
        masks = rearrange(masks,'h w c -> c h w')
    senmatic_map = F.interpolate(senmatic_map.unsqueeze(0).float(),out_size,mode='bilinear',align_corners=False)[0] # c h w
    masks = F.interpolate(masks.unsqueeze(0).float(),out_size,mode='bilinear',align_corners=False)[0] # c h w
    mask_pooling = torch.einsum('chw,mhw->cm',masks,senmatic_map)
    mask_labels = mask_pooling.argmax(-1)
    senmatic_map = senmatic_map.argmax(0)
    masks = masks.argmax(0)
    senmatic_map_voting = torch.zeros_like(senmatic_map,device=senmatic_map.device).long()
    for seg_id in range(len(mask_labels)):
        senmatic_map_voting[masks==seg_id] = mask_labels[seg_id]
    return dict(
        mask_labels=mask_labels,
        senmatic_map_voting=senmatic_map_voting,
        mask_raw=masks,
        senmatic_map_raw=senmatic_map,
    )


def setup_cfg(args,weight):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_hipie_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.task == "grounding":
        cfg.DATASETS.TEST = ("refcoco-unc-val", )
    elif args.task == "detection":
        cfg.DATASETS.TEST = ("coco_2017_val", )
    cfg.MODEL.WEIGHTS = weight
    #cfg.freeze()
    return cfg

def init_demo(cfg,metadata=None):
    demo = VisualizationDemo(cfg,metadata)
    model = demo.predictor.model
    model.enable_clip = False
    model.use_bg_for_pano = True
    model.clip_alpha = 0.2
    model.clip_beta = 0.45
    model.object_mask_threshold = 0.01
    model.overlap_threshold = 0.8
    model.pano_temp_fg = 0.06
    model.pano_temp = 0.06
    return demo,model

def show_anns(anns):
    metadata1=MetadataCatalog.get('coco_2017_train_panoptic_with_sem_seg')
    metadata2=MetadataCatalog.get('ade20k_panoptic_val')
    colors = metadata1.stuff_colors+metadata2.stuff_colors
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    c = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.8]])
        #color_mask = colors[c]+[0.8]
        img[m] = color_mask
        c+=1
    ax.imshow(img)
    
def get_args_eval(name_short='pascal_parts_pano'):
    meta_data_key = dict(
        coco_panoptic='coco_2017_train_panoptic_with_sem_seg',
        ade20k_150='ade20k_panoptic_val',
        ade20k_847='ade20k_full_sem_seg_val',
        pascal_context_59='ctx59_sem_seg_val',
        pascal_context_459='ctx459_sem_seg_val',
        pascal_voc_21='pascal21_sem_seg_val',
        pascal_parts_pano='pascal_parts_val',
        # TODO: Fix pascal
    ) # this is just for reference
    #name = 'ade20k_panoptic_val'
    #name_short = 'coco_panoptic'
    # or
    #name_short = 'coco_panoptic'
    name = meta_data_key[name_short]
    metadata =  MetadataCatalog.get(name)
    cat2ind = cat2ind_panoptics_coco(get_openseg_labels(name_short),name) 
    thing_class_ids = metadata.thing_dataset_id_to_contiguous_id.values()
    is_thing = {k: (k-1 in thing_class_ids )for k,v in cat2ind.items() }
    is_thing[0] = False # BG
    open_seg_labels = get_openseg_labels(name_short,prompt_engineered=True)
    open_seg_labels_no_prompt = get_openseg_labels(name_short,prompt_engineered=False)
    test_args = dict(
        test_categories=open_seg_labels_no_prompt, # for BERT, dont use ensemble
        open_seg_labels=open_seg_labels,
        test_is_thing=is_thing,
        dataset_name='coco_panoptic'
    )
    return test_args


def remap(labelmap):
    labelmap = labelmap + 1
    out = np.zeros_like(labelmap,dtype=labelmap.dtype)
    label_group = MetadataCatalog.get('pascal_parts_val').get('label_group')
    for uuid in np.unique(labelmap):
        if uuid in label_group:
            out[labelmap==uuid] = label_group[uuid]
    out -= 1
    n_cls = len(np.unique(list(MetadataCatalog.get('pascal_parts_val').get('label_group').values())))
    out[out<0] =n_cls
    return out

def sem_to_instance_map(panoptic_seg,segments_info,parts_seg,test_args,max_id=57):
    msks = []
    labels = []
    seg_dict = {x['id']:x['category_id'] for x in segments_info}
    classes = MetadataCatalog.get('pascal_parts_merged_val').stuff_classes if max_id==57  else MetadataCatalog.get('pascal_parts_val').stuff_classes 
    classes_pano =  [x['name'] for x in test_args['test_categories']]
    for v in parts_seg.unique():
        if v == max_id:
            continue
        msk = parts_seg == v
        panoptic_seg_m = panoptic_seg * msk
        uuids = list([int(x) for x in panoptic_seg_m.unique() if x != 0])
        for uuid in uuids:
            panoptic_seg_msk = msk * (panoptic_seg==uuid)
            if panoptic_seg_msk.sum() > 100:
                name = classes[v].split(' ',1)
                if len(name )== 1 or not name[1]:
                    continue
                if name[0] != classes_pano[seg_dict[uuid]]:
                    continue
                name = name[1] if name[1] != 'body' else name[0]
                msks.append(panoptic_seg_msk)
                labels.append(name)
    return msks,labels

def merge_part_and_pano(parts_seg_instance,parts_seg_instance_cls,panoptic_seg,test_args,segments_info):
    parts_seg_instance_all = torch.stack(parts_seg_instance).max(0)[0]
    masks = []
    labels = [] # merged
    panoptic_seg_vv = panoptic_seg.clone()
    seg_dict = {x['id']:x['category_id'] for x in segments_info}
    classes_pano =  [x['name'] for x in test_args['test_categories']]
    for v in panoptic_seg_vv.unique():
        if v == 0:
            continue
        msk = panoptic_seg_vv == v
        area = (msk * parts_seg_instance_all).sum() / msk.sum()
        if area < 0.8:
            masks.append(msk)
            labels.append(classes_pano[seg_dict[int(v)]])
    for msk,text in zip(parts_seg_instance,parts_seg_instance_cls):
        masks.append(msk)
        labels.append(text)
    return masks,labels