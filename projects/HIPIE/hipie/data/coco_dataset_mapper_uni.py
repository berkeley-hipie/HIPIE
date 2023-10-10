# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
import re

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import random
from transformers import AutoTokenizer
from collections import defaultdict
from transformers import RobertaTokenizerFast
from fvcore.transforms.transform import HFlipTransform
from panopticapi.utils import rgb2id
from detectron2.structures import BitMasks, Boxes, Instances
from .objects365_v2 import categories as OBJECTS365V2_CATEGORIES
from .datasets.catids import get_openseg_labels
from .datasets.register_coco_panoptic_annos_semseg import get_metadata
import cv2
from .datasets.register_seginw import _CATEGORIES as SEG_IN_W_CATEGORIES
from .datasets.register_odinw import _CATEGORIES as OD_IN_W_CATEGORIES
COCO_PANOPTIC_META = get_metadata()
__all__ = ["DetrDatasetMapper"]


def cat2ind(categories):
    ind_to_class = {0: '__background__'}
    index = 1
    for x in categories:
        isthing = x["isthing"] if "isthing" in x else 1
        if isthing == 1:
            ind_to_class[index] = x["name"]
            index += 1
    return ind_to_class

def cat2ind_panoptics_coco(categories,name='coco_panoptic'):
    ind_to_class = {0: '__background__'}
    index = 1
    print(f"Length of {name} Categories :",len(categories))
    for x in categories:
        if x['name'] != "invalid_class_id":
            ind_to_class[index] = x["name"]
            index += 1
    return ind_to_class



def create_queries_and_maps(categories, tokenizer, separation_tokens=". ",things_only=False):
    label_list = []
    for x in categories:
        isthing = x["isthing"] if "isthing" in x else 1
        if isthing or (not things_only):
            label_list.append(x["name"])
    labels = list(range(1, len(label_list) + 1)) # [1, 2, ..., 80]

    # Clean label list
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""

    # sep between tokens, follow training
    separation_tokens = ". "
    
    for _index, label in enumerate(label_list):
        
        start_i = len(objects_query)

        objects_query += label
        
        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens

    # print(objects_query) # 'person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush'

    tokenized = tokenizer(objects_query, return_tensors="pt")

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(tokenized, tokens_positive, labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token

def create_queries_and_maps_pan(categories, tokenizer, separation_tokens=". "):
    label_list = []
    for x in categories:
        if x['name'] != "invalid_class_id":
        #if isthing == 1:
            label_list.append(x["name"])
    labels = list(range(1, len(label_list) + 1)) # [1, 2, ..., 80]

    # Clean label list
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""

    # sep between tokens, follow training
    separation_tokens = ". "
    
    for _index, label in enumerate(label_list):
        
        start_i = len(objects_query)

        objects_query += label
        
        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens

    # print(objects_query) # 'person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush'

    tokenized = tokenizer(objects_query, return_tensors="pt")

    # Create the mapping between tokenized sentence and the original label
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(tokenized, tokens_positive, labels=labels)  # from token position to original label
    return objects_query, positive_map_label_to_token


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens

# Unified DataMapper for image-level tasks

SEGM_DATASETS = ["coco_panoptic","ade20k_150",
                    "ade20k_847",
                    "pascal_context_59",
                    "pascal_context_459",
                    "pascal_voc_21","pascal_parts_pano","pascal_parts_merged",
                    "cityscapes_panoptic_parts","paco"]

def filter_instances(instances,num_x):
    if len(instances) <= num_x:
        return instances
    else:
        idx = np.random.choice(range(len(instances)),num_x,replace=False)
        return instances[idx]
class DetrDatasetMapperUni:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def register_segm_dataset(self,name):
            '''
                    "ade20k_150",
                    "ade20k_847",
                    "coco_panoptic",
                    "pascal_context_59",
                    "pascal_context_459",
                    "pascal_voc_21",
                    "lvis_1203",
            '''
            meta_data_key = dict(
                coco_panoptic='coco_2017_train_panoptic_with_sem_seg',
                ade20k_150='ade20k_panoptic_val',
                ade20k_847='ade20k_full_sem_seg_val',
                pascal_context_59='ctx59_sem_seg_val',
                pascal_context_459='ctx459_sem_seg_val',
                pascal_voc_21='pascal21_sem_seg_val',
                pascal_parts_pano='pascal_parts_val',
                pascal_parts_merged='pascal_parts_merged_val',
                cityscapes_panoptic_parts="cityscapes_panoptic_parts_val",
                paco="paco_lvis_val"# TODO: Fix pascal
            )
            assert name in meta_data_key, f"Invalid dataset {name}"
            self.ind_to_class_dict[name] = cat2ind_panoptics_coco(get_openseg_labels(name),name) 
            #self.is_thing[name] = {k: (v in thing_classes )for k,v in self.ind_to_class_dict[name].items() }
            meatdata =  MetadataCatalog.get(meta_data_key[name])
            thing_class_ids = meatdata.thing_dataset_id_to_contiguous_id.values()
            self.is_thing[name] = {k: (k-1 in thing_class_ids )for k,v in self.ind_to_class_dict[name].items() }
            self.is_thing[name][0] = False

    def register_seginw_dataset(self,cat,prefix='seginw'):
            '''
                    "ade20k_150",
                    "ade20k_847",
                    "coco_panoptic",
                    "pascal_context_59",
                    "pascal_context_459",
                    "pascal_voc_21",
                    "lvis_1203",
            '''
            name = f'{prefix}_{cat}'
            open_seg_labels = get_openseg_labels(name)
            self.ind_to_class_dict[name] = cat2ind_panoptics_coco(open_seg_labels,name) 
            #self.is_thing[name] = {k: (v in thing_classes )for k,v in self.ind_to_class_dict[name].items() }
            self.is_thing[name] = {k: True for k in range(len(open_seg_labels)+1) }
            self.open_seg_labels[name] = open_seg_labels
            prompt_test, positive_map_label_to_token = create_queries_and_maps(open_seg_labels, self.tokenizer)
            self.prompt_test_dict[name] = prompt_test
            self.positive_map_label_to_token_dict[name] = positive_map_label_to_token

    def __init__(self, cfg, is_train=True, test_categories=None):
        # test_categories: categories to detect during testing
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeScale(min_scale=0.1,max_scale=2.0,target_height=cfg.INPUT.CROP_SIZE,target_width=cfg.INPUT.CROP_SIZE),
                T.FixedSizeCrop( crop_size=(cfg.INPUT.CROP_SIZE,cfg.INPUT.CROP_SIZE)),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.max_instances = cfg.MODEL.MAX_INSTANCES
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.seg_format = "L"
        
        # language-guided detection
        self.lang_guide_det = cfg.MODEL.LANG_GUIDE_DET
        self.clip_train = cfg.MODEL.CLIP.ENABLED_TRAIN
        if self.lang_guide_det:
            self.ind_to_class_dict = {}
            self.is_thing = {}
            self.prompt_test_dict = {}
            self.positive_map_label_to_token_dict = {}
            self.open_seg_labels = {}
            self.ind_to_class_dict["coco"] = cat2ind(COCO_CATEGORIES)
            self.ind_to_class_dict["obj365v2"] = cat2ind(OBJECTS365V2_CATEGORIES)
            self.ind_to_class_dict['coco_panoptic'] = cat2ind_panoptics_coco(get_openseg_labels("coco_panoptic")) 
            self.is_thing["coco"] = {k:True for k in self.ind_to_class_dict["coco"]}
            self.is_thing["obj365v2"] = {k:True for k in self.ind_to_class_dict["obj365v2"]}
            self.is_thing["coco_panoptic"] = {k:True for k in self.ind_to_class_dict["obj365v2"]}
            use_roberta = cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "roberta-base" and cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "roberta-base"
            if use_roberta:
                self.tokenizer = RobertaTokenizerFast.from_pretrained('projects/HIPIE/roberta-base')
            else:
                self.tokenizer = AutoTokenizer.from_pretrained('projects/HIPIE/bert-base-uncased') # align with GLIP
            for dataset_name in SEGM_DATASETS:
                self.register_segm_dataset(dataset_name)
            for cat in SEG_IN_W_CATEGORIES:
                self.register_seginw_dataset(cat)
            for cat in OD_IN_W_CATEGORIES:
                self.register_seginw_dataset(cat,prefix='odinw')
            self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
            self.part_mode = cfg.MODEL.PART_MODE
            self.prepare = ConvertCocoPolysToMask(
                return_tokens=True,
                tokenizer=self.tokenizer,
                max_query_len=self.max_query_len,
                part_mode = self.part_mode
            )
            if test_categories is not None:
                prompt_test, positive_map_label_to_token = create_queries_and_maps(test_categories, self.tokenizer) # for example, test_categories = [{"name": "person"}]
            else:
                for dataset_name in SEGM_DATASETS:
                    open_seg_labels = get_openseg_labels(dataset_name)
                    self.open_seg_labels[dataset_name] = open_seg_labels
                    prompt_test, positive_map_label_to_token = create_queries_and_maps(open_seg_labels, self.tokenizer)
                    self.prompt_test_dict[dataset_name] = prompt_test
                    self.positive_map_label_to_token_dict[dataset_name] = positive_map_label_to_token
                #COCO,O365
                prompt_test, positive_map_label_to_token = create_queries_and_maps(COCO_CATEGORIES, self.tokenizer,things_only=True)
                self.prompt_test_dict["coco"] = prompt_test
                self.positive_map_label_to_token_dict["coco"] = positive_map_label_to_token
                prompt_test, positive_map_label_to_token = create_queries_and_maps(OBJECTS365V2_CATEGORIES, self.tokenizer)
                self.prompt_test_dict["obj365v2"] = prompt_test
                self.positive_map_label_to_token_dict["obj365v2"] = positive_map_label_to_token
        # oridinal numbers
        self.ordinal_nums = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        self.always_true = {k:True for k in range(200)} # TO BE SAFE
    def transform_img(self, image, disable_crop=False,sem_seg=None):
        if sem_seg is not None:
            image = T.AugInput(image,sem_seg=sem_seg)
            

        if self.crop_gen is None or disable_crop:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )
        if sem_seg is not None:
            image, sem_seg_gt = image.image, image.sem_seg

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        if sem_seg is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            return (image,sem_seg_gt), image_shape, transforms
        else:
            return image, image_shape, transforms
    
    def transform_expressions(self, expressions, transforms):
        # pick one expression if there are multiple expressions
        expression = expressions[np.random.choice(len(expressions))]
        expression = clean_string(expression)
        # deal with hflip for expression
        hflip_flag = False
        for x in transforms:
            if isinstance(x, HFlipTransform):
                hflip_flag = True
                break
        if hflip_flag:
            expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')
        return expression

    def transform_annos(self, annotations_ori, transforms, image_shape, dataset_dict,overwrite_instances=None):
        # USER: Implement additional transformations if you have other types of data
        
        if overwrite_instances is not None:
            instances = overwrite_instances
        else:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in annotations_ori
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        # language-guided detection
        task = dataset_dict["task"] if "task" in dataset_dict else None
        if self.lang_guide_det and task == "detection":
            ind_to_class = self.ind_to_class_dict[dataset_dict["dataset_name"]]
            is_thing =  self.is_thing[dataset_dict["dataset_name"]]
            original_box_num = len(instances)
            instances, positive_caption_length = check_for_positive_overflow(instances, ind_to_class, self.tokenizer, self.max_query_len-2)
            if len(instances) < original_box_num:
                print("WARNING: removed {} boxes due to positive caption overflow".format(original_box_num - len(instances)))
            if not self.clip_train:
                annotations, caption, label_to_positions = convert_object_detection_to_grounding_optimized_for_od(
                    instances=instances, ind_to_class=ind_to_class,
                    positive_caption_length=positive_caption_length,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_query_len-2
                )
            else:
                annotations, caption, label_to_positions = convert_object_detection_to_grounding_optimized_for_od(
                    instances=instances, ind_to_class=ind_to_class,
                    disable_shuffle=True,
                    control_probabilities=(0.0, 0.0, 1.0, 0.0),
                    positive_caption_length=positive_caption_length,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_query_len-2
                )
            anno = {"annotations": annotations, "caption": caption, "label_to_positions": label_to_positions}
            anno = self.prepare(anno)
            instances.positive_map = anno["positive_map"].bool() # (N, max_seq_len). N is num of objects. bool() -> 0 or 1
            if self.clip_train:
                labels = [[y for y in x.strip().split(',')] for x in caption.split('.')]
                names = [ind_to_class[y+1] for y in instances.gt_classes.numpy()] 
                start = 0
                all_labs = {}
                for idx,a in enumerate(labels):
                    all_labs[','.join(a)] = (start,start+len(a))
                    start += len(a)
                positive_maps = torch.zeros(anno["positive_map"].shape[0],start)
                for idx,tt in enumerate(names):
                    start,end = all_labs[tt]
                    positive_maps[idx][start:end] = 1.0
                instances.positive_map = positive_maps.bool()

            instances.is_thing = torch.tensor([is_thing[x.item()+1] for x in instances.gt_classes]).bool()
            expressions_new = anno["caption"] # "expressions" are shared between detection and grounding
            # Check FLIP
            hflip_flag = False
            for x in transforms:
                if isinstance(x, HFlipTransform):
                    hflip_flag = True
                    break
            if hflip_flag:
                expressions_new = expressions_new.replace('left', '@').replace('right', 'left').replace('@', 'right')
            
            #expressions_new = self.transform_expressions(expressions_new,transforms)
        elif self.lang_guide_det and task == "grounding":
            instances.positive_map = torch.ones((1, 1), dtype=torch.bool) # 1 instance, 1 (pooled) token.
            instances.is_thing = torch.tensor([True], dtype=torch.bool)
            expressions_new = dataset_dict["expressions"]
        elif self.lang_guide_det and task == "phrase_grounding":
            expressions_new = dataset_dict["expressions"]
            anno = {"annotations": dataset_dict["annotations"], "caption": expressions_new}
            anno = self.prepare(anno)
            instances.positive_map = anno["positive_map"].bool() # (N, max_seq_len). N is num of objects. bool() -> 0 or 1
            instances.is_thing = torch.tensor([True]*len(anno["positive_map"]), dtype=torch.bool)
            expressions_new = anno["caption"] # "expressions" are shared between detection and grounding
        else:
            raise ValueError("task must be detection or grounding")
        if hasattr(instances, "gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        return instances, expressions_new

    def has_ordinal_num(self, expressions_list):
        flag = False
        for expression in expressions_list:
            expression_low = expression.lower()
            for word in self.ordinal_nums:
                if word in expression_low:
                    flag = True
                    break
            if flag == True:
                break
        return flag

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Panoptic extension
        # USER: Remove if you don't do semantic/panoptic segmentation
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), format=self.seg_format
            )
            if self.seg_format == "L":
                sem_seg_gt = sem_seg_gt.squeeze(2)
        else:
            sem_seg_gt = None


        # USER: Modify this if you want to keep them for some reason.
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)
        # if there are ordinal numbers in expressions, disable crop
        disable_crop = self.has_ordinal_num(dataset_dict["expressions"]) if "expressions" in dataset_dict else False
        transformed_image, image_shape, transforms = self.transform_img(image, disable_crop=disable_crop,sem_seg=sem_seg_gt)
        if sem_seg_gt is not None:
            (dataset_dict["image"],dataset_dict["sem_seg"]) = transformed_image
        else:
            dataset_dict["image"] = transformed_image
        if 'pan_seg_file_name' in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
            pan_seg_gt = rgb2id(pan_seg_gt)
            dataset_dict["pan_seg_gt"] = torch.from_numpy(np.ascontiguousarray(pan_seg_gt))
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])
            classes = np.array(classes)
            if len(masks) == 0:
                gt_masks = BitMasks(torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])))
                gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                gt_masks = masks
                gt_boxes = masks.get_bounding_boxes()
            instances = Instances(image_shape)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_masks = gt_masks
            instances.gt_boxes = gt_boxes
        elif dataset_dict.get('dataset_name',None) in ['pascal_parts_pano','pascal_parts_merged','cityscapes_panoptic_parts']:
            instances = Instances(image_shape)
            classes = []
            masks = []
            sem_seg_gt_transformed = transforms.apply_segmentation(sem_seg_gt)
            for uuid in np.unique(sem_seg_gt_transformed):
                if uuid == 0:
                    continue
                binary_mask = sem_seg_gt_transformed==uuid
                msk = binary_mask
                #print(binary_mask.shape)
                num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8) * 255)
                #print(num_labels,labels.min(),labels.max())
                # for i in range(num_labels):
                #     msk = labels==i
                #     msk = msk & binary_mask
                #     area = msk.sum()
                #     if area < 100:
                #         continue
                    #print(area)
                classes.append(uuid-1)
                masks.append(msk)
            if len(masks) == 0:
                gt_masks = BitMasks(torch.zeros((0, sem_seg_gt_transformed.shape[-2], sem_seg_gt_transformed.shape[-1])))
                gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                gt_masks = masks
                gt_boxes = masks.get_bounding_boxes()
            instances = Instances(image_shape)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            instances.gt_masks = gt_masks
            instances.gt_boxes = gt_boxes
        else:
            instances = None

        if "expressions" in dataset_dict and dataset_dict["task"] == "grounding":
            dataset_dict["expressions"] = self.transform_expressions(dataset_dict["expressions"], transforms)
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            # language-guided detection
            task = dataset_dict["task"] if "task" in dataset_dict else None
            if self.lang_guide_det and task == "detection":
                dataset_dict["expressions"] = self.prompt_test_dict[dataset_dict["dataset_name"]]
                dataset_dict["is_thing"] = self.is_thing[dataset_dict["dataset_name"]]
                dataset_dict['open_seg_labels'] = get_openseg_labels(dataset_dict["dataset_name"],prompt_engineered=True)#self.open_seg_labels.get(dataset_dict["dataset_name"])
                dataset_dict["positive_map_label_to_token"] = self.positive_map_label_to_token_dict[dataset_dict["dataset_name"]]
                positive_map_label_to_token_dict = {}
                if self.clip_train:
                    caption = dataset_dict["expressions"]
                    ind_to_class = self.ind_to_class_dict[dataset_dict["dataset_name"]]
                    labels = [[y for y in x.strip().split(',')] for x in caption.split('.')]
                    #names = [ind_to_class[y+1] for y in (len(self.positive_map_label_to_token_dict))] 
                    start = 0
                    all_labs = {}
                    for idx,a in enumerate(labels):
                        all_labs[','.join(a)] = (start,start+len(a))
                        start += len(a)
                    for k in dataset_dict["positive_map_label_to_token"].keys():
                        start,end = all_labs[ind_to_class[k]]
                        positive_map_label_to_token_dict[k] = list(range(start,end))
                    dataset_dict["positive_map_label_to_token"] = positive_map_label_to_token_dict
                    
            else:
                dataset_dict["is_thing"] = self.always_true
            return dataset_dict

        if "annotations" in dataset_dict or instances is not None:
            instances, expressions_new = self.transform_annos(dataset_dict.get("annotations",None), transforms, image_shape, dataset_dict,overwrite_instances=instances)
            # add "expressions" for detection data
            dataset_dict["expressions"] = expressions_new
            instances = utils.filter_empty_instances(instances)
            if self.max_instances > 0:
                instances = filter_instances(instances,self.max_instances)
            if len(instances) == 0:
                return None 
            dataset_dict["instances"] = instances
        if dataset_dict["task"] == "phrase_grounding":
            dataset_dict["task"] = "detection"
        return dataset_dict

# generate image pairs (reference-key) based on still images
# This mapper is used only for training
class DetrDatasetMapperUniCLIP(DetrDatasetMapperUni):
    def __init__(self, cfg, is_train=True, test_categories=None):
        super().__init__(cfg, is_train=is_train, test_categories=test_categories)
        assert self.is_train
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image_key = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image_key)

        # USER: Modify this if you want to keep them for some reason.
        dataset_dict["image"] = []
        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            anno.pop("keypoints", None)
        annotations_key = dataset_dict.pop("annotations")
        annotations_ref = copy.deepcopy(annotations_key)
        image_ref = copy.deepcopy(image_key)
    
        image_key, image_shape_key, transforms_key = self.transform_img(image_key)
        dataset_dict["image"].append(image_key)
        image_ref, image_shape_ref, transforms_ref = self.transform_img(image_ref)
        dataset_dict["image"].append(image_ref)
        assert "expressions" not in dataset_dict

        dataset_dict["expressions"] = []
        instances_key, expressions_new_key = self.transform_annos(annotations_key, transforms_key, image_shape_key, dataset_dict)
        instances_ref, expressions_new_ref = self.transform_annos(annotations_ref, transforms_ref, image_shape_ref, dataset_dict)
        # add "expressions" for detection data
        dataset_dict["expressions"].append(expressions_new_key)
        dataset_dict["expressions"].append(expressions_new_ref)


        instances_key_tmp = utils.filter_empty_instances(copy.deepcopy(instances_key))
        instances_ref_tmp = utils.filter_empty_instances(copy.deepcopy(instances_ref))
        if len(instances_key_tmp) == 0 or len(instances_ref_tmp) == 0:
            return None 
        _gt_ids = list(range(1,1+len(instances_ref)))
        instances_key.gt_ids = torch.tensor(_gt_ids)
        instances_ref.gt_ids = torch.tensor(_gt_ids)
        dataset_dict["instances"] = [filter_empty_instances_soft(instances_key),  filter_empty_instances_soft(instances_ref)] # instances of two frames
        # for key/ref frame， we don't remove empty instances，but mark them with gt_ids=-1, and process them in idol.py
        # gt_ids has no practical meaning, we just use it as a flag to indicate whether an instance exists, 
        # idx indicates the object correspondence between key&reference frame

        return dataset_dict

def filter_empty_instances_soft(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1 # invalid instances are marked with -1
    return instances

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')

def check_for_positive_overflow(instances, ind_to_class, tokenizer, max_seq_length=256):
    # NOTE: Only call this function for OD data; DO NOT USE IT FOR GROUNDING DATA
    # NOTE: called only in coco_dt

    # Check if we have too many positive labels
    # generate a caption by appending the positive labels
    positive_label_set = set()
    for i in range(len(instances)):
        label_i = instances.gt_classes[i].item() + 1 # "+1" for mapping 0~79 to 1~80
        positive_label_set.add(label_i)
    positive_label_list = list(positive_label_set)

    # random shuffule so we can sample different annotations at different epochs
    random.shuffle(positive_label_list)

    kept_lables = []
    length = 0

    for index, label in enumerate(positive_label_list):

        label_text = clean_name(ind_to_class[label]) + ". " # "dog. "

        tokenized = tokenizer.tokenize(label_text)

        length += len(tokenized)

        if length > max_seq_length: # there could not be overflow for COCO dataset
            break
        else:
            kept_lables.append(label)
    
    ## filter boxes
    keep_box_index = []
    for i in range(len(instances)):
        label_i = instances.gt_classes[i].item() + 1 # "+1" for mapping 0~79 to 1~80
        if label_i in kept_lables:
            keep_box_index.append(i)
    
    # keep_box_index = torch.LongTensor(keep_box_index)
    instances = instances[keep_box_index] ## filter boxes

    return instances, length

def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name

def convert_object_detection_to_grounding_optimized_for_od(
        instances,
        ind_to_class,
        disable_shuffle=False,
        add_detection_prompt=False,
        add_detection_prompt_advanced=False,
        random_sample_negative=85,
        control_probabilities=(0.0, 0.0, 0.5, 0.0),
        restricted_negative_list=None,
        separation_tokens=". ",
        max_num_labels=-1,
        max_seq_length=256,
        tokenizer=None,
        positive_caption_length=0
):
    '''
    ind_to_class: {0: "__background__", 1 : "person" ...}
    instances:

    restricted_negative_list : for datasets with restricted negatives, sample only the negatives

    Convert object detection data into grounding data format, on the fly.

    Control options:
        1. add_detection_prompt: add "object detection : " to the front of the prompt
        2. num_negatives: randomly sampled negative classes
        3. num_positives: how many positives to keep (-1 means do not cut any)

    Probabilities to generate the control options:

        a. probability_one_negative: only give one negative class to mimic evaluation
        b. probability_one_positive: only give one positive class to mimic evaluation
        c. probability_full: add both all positive and all negatives
        d. other:
            randomly sample some negatives and some positives
            The below control options are independent of each other:
            - probability_random_negative: probability of randomly sample X negatives
            - probability_random_positive: probability of randomly sample some positives
    '''
    if restricted_negative_list is None: # True
        valid_negative_indexes = list(ind_to_class.keys()) # [0, 1, 2, ... 80]
    else:
        valid_negative_indexes = restricted_negative_list
    # import ipdb; ipdb.set_trace()
    def generate_senetence_given_labels(
            positive_label_list,
            negative_label_list,
            prompt_engineer_version="v2",
            disable_shuffle=False):

        '''
        v3: with simple prompt such as "there are", "are there?"
        v4: try to merge some are there / there are together, to avoid sequence being too long
        '''

        label_to_positions = {}

        assert (prompt_engineer_version == "v2")
        num_negatives = len(negative_label_list)
        num_positives = len(positive_label_list)
        label_list = negative_label_list + positive_label_list
        if not disable_shuffle: # True
            random.shuffle(label_list)

        if add_detection_prompt: # False
            if add_detection_prompt_advanced and (num_negatives == 0 or num_positives == 0) and not disable_shuffle:
                pheso_caption = "object detection query : "
            else:
                pheso_caption = "object detection : "
        else:
            pheso_caption = ""

        for index, label in enumerate(label_list):

            start_index = len(pheso_caption)

            pheso_caption += clean_name(ind_to_class[label])  # NOTE: slight change...
            end_index = len(pheso_caption)

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens # += ". "

        return label_to_positions, pheso_caption

    if disable_shuffle: # False
        label_list = list(sorted(ind_to_class.keys()))[1:]  # do not include the background
        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=label_list,
            negative_label_list=[],
            disable_shuffle=True)
        # print(label_to_positions, pheso_caption)
    else:
        positive_label_set = set()
        for i in range(len(instances)):
            label_i = instances.gt_classes[i].item() + 1
            positive_label_set.add(label_i)

        full_positive = len(positive_label_set) # num classes containing in the current image
        if max_num_labels <= 0: # -1
            full_negative = random_sample_negative # 85
        else:
            full_negative = max(min(max_num_labels-full_positive, random_sample_negative), 0)

        if full_negative > len(valid_negative_indexes): # True (85 > 81)
            full_negative = len(valid_negative_indexes) # 81

        num_negatives, num_positives = generate_control_options_given_probabilities(
            control_probabilities=control_probabilities, # (0.0, 0.0, 0.5, 0.0)
            full_positive=full_positive,
            full_negative=full_negative)
        # num_positives not used
        

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)
            for i in np.random.choice(valid_negative_indexes, size=num_negatives, replace=False):
                # label_sets.add(i)
                if i not in positive_label_set:
                    negative_label_list.add(i)

        # Keep all positives; ignoring num_positives
        positive_label_list = list(positive_label_set)
        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)  # e.g.: [17, 1, 13] where each number is the class name
        random.shuffle(negative_label_list)

        # Do a pre-screen. If we cannot afford this many negatives, we will sample less
        negative_max_length = max_seq_length - positive_caption_length
        screened_negative_label_list = []
        for negative_label in negative_label_list:
            label_text = clean_name(ind_to_class[negative_label]) + ". " # "dog. "

            tokenized = tokenizer.tokenize(label_text)
            
            negative_max_length -= len(tokenized)

            if negative_max_length > 0: 
                screened_negative_label_list.append(negative_label) # keep this negative
            else:
                break
        negative_label_list = screened_negative_label_list

        label_to_positions, pheso_caption = generate_senetence_given_labels(
            positive_label_list=positive_label_list,
            negative_label_list=negative_label_list)
    new_target = []
    # label_to_positions: dict
    # key: class index (range from 0-80)
    # value: their (char-level) positions in the caption
    for i in range(len(instances)):
        new_target_i = {}
        label_i = instances.gt_classes[i].item() + 1
        if label_i in label_to_positions:  # NOTE: Only add those that actually appear in the final caption
            new_target_i["tokens_positive"] = [label_to_positions[label_i]]
            new_target.append(new_target_i)
    return new_target, pheso_caption, label_to_positions


def generate_control_options_given_probabilities(
        control_probabilities,
        full_positive,
        full_negative):
    
    # The function was originally designed to perform data augmentation by randomly dropping negative and positive classes. Later, we decided to only consider dropping negative classes. So the returned 'num_positives' by this function will be ignored.
    # 0828 use all positive classes. prob 0.5 -> use all negative classes; prob 0.5 -> random number of negative classes
    outer_prob = random.random()

    probability_one_negative = control_probabilities[0]
    probability_one_positive = control_probabilities[1]
    probability_full = control_probabilities[2] # 0.5
    probability_drop_positive = control_probabilities[3]

    assert(probability_drop_positive == 0)

    if outer_prob < probability_one_negative:
        # a. probability_one_negative: only give one negative class to mimic evaluation (10%)
        num_negatives = 1
        num_positives = 0
    elif outer_prob < probability_one_positive + probability_one_negative:
        # b. probability_one_positive: only give one positive class to mimic evaluation (10%)
        num_negatives = 0
        num_positives = 1
    elif outer_prob < probability_full + probability_one_positive + probability_one_negative: # prob 0.5
        # c. probability_full: add both all positive and all negatives (20%)
        num_negatives = full_negative
        num_positives = full_positive
    else: # prob 0.5
        if random.random() < 1.0:  # - probability_random_negative: probability of randomly sample X negatives (100%)
            num_negatives = np.random.choice(max(1, full_negative)) + 1  # mininum 1
        else:
            num_negatives = full_negative  # Full

        if random.random() < probability_drop_positive:  # False
            num_positives = np.random.choice(max(1, full_positive)) + 1
        else:
            num_positives = full_positive  # Full

    return num_negatives, num_positives

class ConvertCocoPolysToMask(object):
    def __init__(self, return_tokens=False, tokenizer=None, max_query_len=256,part_mode=False):
        self.return_tokens = return_tokens # True
        self.tokenizer = tokenizer # AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_query_len = max_query_len
        self.part_mode = part_mode

    def __call__(self, target):

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None
        tokens_positive = [obj["tokens_positive"] for obj in anno]

        target = {}
        if caption is not None:
            target["caption"] = caption

        if tokens_positive is not None:
            target["tokens_positive"] = tokens_positive

        if self.return_tokens and self.tokenizer is not None: # True
            tokenized = self.tokenizer(caption, return_tensors="pt",
                max_length=self.max_query_len,
                truncation=True)
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"],self.max_query_len,self.tokenizer,self.part_mode) # (N, 256) N is num of objects. value > 0 where positive class
            # if a class name is tokenized into M tokens, value is 1/M. For example, if a class name is divided into 3 tokens, value is 1/3

        return target

def create_positive_map(tokenized, tokens_positive,max_seq_len=256,tokenizer=False,part_mode=True):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), max_seq_len), dtype=torch.float)
    #print(tokenized,tokens_positive,tokenized['input_ids'].shape)
    input_ids = tokenized['input_ids']
    assert input_ids.shape[0] == 1
    unique_ids,unique_id_counts = input_ids.unique(return_counts=True)

    for j, tok_list in enumerate(tokens_positive): # loop over each object
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    if tokenizer and part_mode:
        input_ids = input_ids[0]
        input_ids = torch.cat([input_ids,torch.zeros(max_seq_len-len(input_ids),dtype=int)-1])
        decoded_tokens = tokenizer.batch_decode(unique_ids[:,None]) # L
        for uid,uid_c,uid_str in zip(unique_ids,unique_id_counts,decoded_tokens):
            if '[' in uid_str or ']' in uid_str or '#' in uid_str or '.' in uid_str or uid_c == 1 or uid_c >=30 or uid_str in ['lower','upper']:
                continue
            matched_ids = input_ids == uid
            #print(matched_ids)
            #print(uid,uid_c,positive_map.shape,uid_str)
            #print(positive_map[:,matched_ids].sum())
            positive_map[:,matched_ids] = torch.any(positive_map[:,matched_ids],-1,keepdim=True).float()
            #print(positive_map[:,matched_ids].sum())
            #print('====')
        #print(decoded_tokens,positive_map.shape,positive_map)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j, iff token i is mapped to j label"""
    positive_map = defaultdict(int)

    # Additionally, have positive_map_label_to_tokens
    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = [] 
            for i in range(beg_pos, end_pos + 1):
                positive_map[i] = labels[j]  # because the labels starts from 1
                positive_map_label_to_token[labels[j]].append(i)
            # positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map, positive_map_label_to_token  # / (positive_map.sum(-1)[:, None] + 1e-6)
