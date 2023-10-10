import os

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys

# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.projects.hipie import add_hipie_config
from projects.HIPIE.predictor import VisualizationDemo
from argparse import Namespace

from detectron2.data import MetadataCatalog,DatasetCatalog,Metadata
from detectron2.projects.hipie.data.coco_dataset_mapper_uni import get_openseg_labels,cat2ind_panoptics_coco,create_queries_and_maps
from detectron2.structures import Instances
from detectron2.projects.hipie.demo_lib.demo_utils import  vote
import torch
from argparse import Namespace
from fairscale.nn.checkpoint import checkpoint_wrapper 

def setup_cfg(args,weight,device=None):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_hipie_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # HIPIE
    if args.task == "grounding":
        cfg.DATASETS.TEST = ("refcoco-unc-val", )
    elif args.task == "detection":
        cfg.DATASETS.TEST = ("coco_2017_val", )
    cfg.MODEL.WEIGHTS = weight
    if device is not None:
        cfg.MODEL.DEVICE = device 
    #cfg.freeze()
    return cfg

def build_standard_demo(
        config_file ='/home/jacklishufan/HIPIE/projects/HIPIE/configs/image_joint_vit_huge_32g_pan_maskdino_ade_test.yaml',
        weight = 'outputs/hipie_output/vit_h_cloud/model_final.pth',
        device=None,
):
    args = Namespace()
    args.config_file = config_file
    args.opts = ['OUTPUT_DIR','outputs/test_r50_maskdino_pan_fixed_lan']
    args.task = "detection"
    cfg = setup_cfg(args,weight,device=device)
    cfg.MODEL.CLIP.ALPHA = 0.2
    cfg.MODEL.CLIP.BETA = 0.45
    cfg.MODEL.PANO_TEMPERATURE_CLIP_FG = 0.01
    cfg.MODEL.PANO_TEMPERATURE = 0.06
    demo = VisualizationDemo(cfg)
    return demo


def get_args_eval(name_short='pascal_parts_pano'):
    from detectron2.data import MetadataCatalog,DatasetCatalog
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
    meatdata =  MetadataCatalog.get(name)
    cat2ind = cat2ind_panoptics_coco(get_openseg_labels(name_short),name) 
    thing_class_ids = meatdata.thing_dataset_id_to_contiguous_id.values()
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

# def sem_to_instance_map(panoptic_seg,segments_info,parts_seg,max_id=57):
#         msks = []
#         labels = []
#         seg_dict = {x['id']:x['category_id'] for x in segments_info}
#         classes = MetadataCatalog.get('pascal_parts_merged_val').stuff_classes if max_id==57  else MetadataCatalog.get('pascal_parts_val').stuff_classes 
#         classes_pano =  [x['name'] for x in test_args['test_categories']]
#         for v in parts_seg.unique():
#             if v == max_id:
#                 continue
#             msk = parts_seg == v
#             panoptic_seg_m = panoptic_seg * msk
#             uuids = list([int(x) for x in panoptic_seg_m.unique() if x != 0])
#             for uuid in uuids:
#                 panoptic_seg_msk = msk * (panoptic_seg==uuid)
#                 if panoptic_seg_msk.sum() > 100:
#                     name = classes[v].split(' ',1)
#                     if len(name )== 1 or not name[1]:
#                         continue
#                     if name[0] != classes_pano[seg_dict[uuid]]:
#                         continue
#                     name = name[1] if name[1] != 'body' else name[0]
#                     msks.append(panoptic_seg_msk)
#                     labels.append(name)
#         return msks,labels

def sem_to_instance_map_by_instances(instance_masks,parts_seg,max_id=57,instance_label_names=None):
        msks = []
        labels = []
        #seg_dict = {x['id']:x['category_id'] for x in segments_info}
        classes = MetadataCatalog.get('pascal_parts_merged_val').stuff_classes if max_id==57  else MetadataCatalog.get('pascal_parts_val').stuff_classes 
        #classes_pano =  [x['name'] for x in test_args['test_categories']]
        for v in parts_seg.unique():
            if v == max_id:
                continue
            msk = parts_seg == v
            for idx,inst_msk in enumerate(instance_masks):
                panoptic_seg_msk = msk * inst_msk
                if panoptic_seg_msk.sum() > 100:
                    name = classes[v].split(' ',1)
                    if len(name )== 1 or not name[1]:
                        continue
                    if instance_label_names is not None and name[0] != instance_label_names[idx]:
                        continue
                    name = name[1] if name[1] != 'body' else name[0]
                    msks.append(panoptic_seg_msk)
                    labels.append(name)
        return msks,labels
# def merge_part_and_pano(parts_seg_instance,parts_seg_instance_cls,panoptic_seg,segments_info):
#         parts_seg_instance_all = torch.stack(parts_seg_instance).max(0)[0]
#         masks = []
#         labels = [] # merged
#         panoptic_seg_vv = panoptic_seg.clone()
#         seg_dict = {x['id']:x['category_id'] for x in segments_info}
#         classes_pano =  [x['name'] for x in test_args['test_categories']]
#         for v in panoptic_seg_vv.unique():
#             if v == 0:
#                 continue
#             msk = panoptic_seg_vv == v
#             area = (msk * parts_seg_instance_all).sum() / msk.sum()
#             if area < 0.8:
#                 masks.append(msk)
#                 labels.append(classes_pano[seg_dict[int(v)]])
#         for msk,text in zip(parts_seg_instance,parts_seg_instance_cls):
#             masks.append(msk)
#             labels.append(text)
#         return masks,labels
HIERARCHAL = {
    "head":[
        "ear","eye","nose","muzzle","horn"
    ]
}

SYN = [
    ["nose","muzzle"]
]
class PartSegmDemo:

    def __init__(self,
                 config_file,
                 weight,device=None) -> None:
        self.demo = build_standard_demo(
            config_file,
            weight,
            device=device
        )
        self.arg_parts = get_args_eval('pascal_parts_pano')
        self.arg_coco =  get_args_eval('coco_panoptic')

    def match(self,query,part_name):
        if query == part_name:
            return True
        if query in part_name:
            return True
        if query in HIERARCHAL and part_name in HIERARCHAL[query]:
            return True
        for arr in SYN:
            if query in arr and part_name in arr:
                return True
        return False

    def get_part_segmentation(self,img):
        predictions, visualized_output = self.demo.run_on_image(img, 0.5,
                "detection", None,**self.arg_parts)
        parts_seg = predictions['sem_seg'].cpu().argmax(0) # H X W  Long 
        remapped = remap(parts_seg.numpy())
        return dict(
            parts_seg=torch.tensor(parts_seg),
            remapped=torch.tensor(remapped),
        )
    

    def retrive_by_points(self,points,masks,return_masks=3):
        all_masks = masks['all_masks'].cpu() > 0
        pts_mas = torch.zeros_like(all_masks[0])
        pts_mas[points[:,1],points[:,0]] = True
        matches = (all_masks * pts_mas[None,]).flatten(1).sum(-1).numpy()
        areas = all_masks.flatten(1).sum(1).numpy()
        matched_points, areas, indices = np.array(sorted([(-x,y,z) for z,(x,y) in enumerate(zip(matches,areas))])).T
        return all_masks[indices[:return_masks]]

    
    def parse_def_string(self,input_str=None):
        if input_str is None:
            input_str = '''
            1:thing:dog
            2:stuff:sky
            3:stuff:grass
            '''
        lines = input_str.splitlines()

        custom_categories = []
        custom_categories_is_thing = {}
        custom_categories_is_thing[0] = False
        cid = 1
        thing_classes = {}
        stuff_classes = {}
        for line in lines:
            if not line:
                continue
            id,is_thing_cls,name = line.split(":")
            custom_categories_is_thing[cid] = is_thing_cls=='thing'
            if name == 'invalid_class_id':
                continue
            custom_categories.append({"id": int(id), "name": name})
            if custom_categories_is_thing[cid]:
                thing_classes[cid-1] = name
            else:
                stuff_classes[cid-1] = name
            cid += 1
        test_args_custom = dict(
            test_categories=custom_categories, # for BERT, dont use ensemble
            open_seg_labels=custom_categories,
            test_is_thing=custom_categories_is_thing,
        )

        meta_data = Metadata(stuff_classes=stuff_classes,
                              thing_classes=thing_classes,
                              stuff_dataset_id_to_contiguous_id={v:v-len(thing_classes) for v in range(100)},
                              things_dataset_id_to_contiguous_id={v:v for v in range(100)},
                              )
        return test_args_custom,meta_data
    
    def foward_panoptic(self,img,things_labels=None,stuff_labels=None,def_string=None,instance_thres=0.5,do_part=False):
        if def_string is None:
            assert things_labels is not None and stuff_labels is not None, "Either a defstring or list of labels is needed"
            def_string = []
            idx = 1
            for x in things_labels:
                def_string.append(f'{idx}:thing:{x}')
                idx += 1
            for x in stuff_labels:
                def_string.append(f'{idx}:stuff:{x}')
                idx += 1
            def_string = '\n'.join(def_string)
        test_args,meta_data = self.parse_def_string(def_string)
        predictions, visualized_output = self.demo.run_on_image(img, 0.5,
                "detection", None,**test_args)
        predictions['panoptic_seg'] = (predictions['panoptic_seg'][0].to('cpu'),predictions['panoptic_seg'][1])
        predictions['instances'] = predictions['instances'].to('cpu')
        valid = predictions['instances'].scores > instance_thres
        new_instances = Instances(predictions['instances']._image_size)
        new_instances.pred_boxes = predictions['instances'].pred_boxes[valid]
        new_instances.pred_masks = predictions['instances'].pred_masks[valid]
        new_instances.pred_classes =  predictions['instances'].pred_classes[valid]
        new_instances.scores =  predictions['instances'].scores[valid]
        predictions['instances'] = new_instances

        if do_part:
            part_output = self.get_part_segmentation(img)
            instance_mask = predictions['instances'].pred_masks.cpu()
            instance_label_names = [meta_data.thing_classes[x] for x in predictions['instances'].pred_classes.cpu().numpy()]
            output_refined = sem_to_instance_map_by_instances(
                    instance_mask,
                    part_output['parts_seg'],max_id=200,instance_label_names=instance_label_names)
            output_coarse = sem_to_instance_map_by_instances(
                instance_mask,
                part_output['remapped'],max_id=57,instance_label_names=instance_label_names)
            predictions.update(
                output_refined=output_refined,
                output_coarse=output_coarse,
            )
        predictions.update(test_args=test_args,meta_data=meta_data)
        return predictions



    def foward_reference(self,img,reference_string,part,use_coarse=False):
        predictions, visualized_output = self.demo.run_on_image(img,
                    0.5,
                'grounding', expressions=reference_string,**self.arg_coco)
        instance_mask = predictions['instances'].pred_masks.cpu()
        out_dict = dict(
            instance_mask=instance_mask
        )
        if part is not None:
            part_output = self.get_part_segmentation(img)
            output_refined = sem_to_instance_map_by_instances(
                instance_mask,
                part_output['parts_seg'],max_id=200)
            output_coarse = sem_to_instance_map_by_instances(
                instance_mask,
                part_output['remapped'],max_id=57)
            out_dict.update(
                output_refined=output_refined,
                output_coarse=output_coarse,
            )
            final_mask = torch.zeros_like(instance_mask,dtype=instance_mask.dtype)
            a,b = (torch.stack(output_coarse[0]),output_coarse[1]) if use_coarse else (torch.stack(output_refined[0]),output_refined[1])
            for part_mask,part_name in zip(
                a,
                b
            ):
                if self.match(part,part_name):
                    final_mask += part_mask
            final_mask = final_mask.clip(min=0,max=1).to(torch.uint8)
        else:
           final_mask=instance_mask
        out_dict.update(final_mask=final_mask)
        return out_dict
