# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from .backbone.masked_backbone import MaskedBackbone
from .models.deformable_detr.backbone import Joiner
from .models.deformable_detr.deformable_detr import DeformableDETR, SetCriterion, DeformableDETRDINO, DINOCriterion
from .models.deformable_detr.matcher import HungarianMatcherVL
from .models.deformable_detr.matcher_mask import HungarianMatcher as HungarianMatcherBG
from .models.deformable_detr.position_encoding import PositionEmbeddingSine
from .models.deformable_detr.deformable_transformer import DeformableTransformerVL
from .models.deformable_detr.deformable_transformer_dino import DeformableTransformerVLDINO
from .models.ddetrs import DDETRSegmUni, segmentation_postprocess
from .models.ddetrs_dn import DDETRSegmUniDN
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detectron2.utils.memory import retry_if_cuda_oom
from .util.misc import NestedTensor
import torchvision.ops as ops
# Language-guided detection
from transformers import AutoTokenizer
from .models.deformable_detr.bert_model import BertEncoder
from .util.misc import NestedTensor,  inverse_sigmoid
from .open_vocab.helper import prompt_labels

from collections import OrderedDict
from einops import repeat
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)
from detectron2.structures import BoxMode
import cv2
from skimage import color
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from .models.sam import build_sam
from .open_vocab.clip import MaskCLIP
from .data.coco_dataset_mapper_uni import get_openseg_labels
__all__ = ["HIPIE_IMG"]

@META_ARCH_REGISTRY.register()
class HIPIE_IMG(nn.Module):
    """
    Unified model for image-level tasks (OD, IS, REC, RES)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.demo_only = False
        self.num_frames = 1
        self.use_amp = cfg.SOLVER.AMP.ENABLED
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.mask_stride = cfg.MODEL.DDETRS.MASK_STRIDE
        self.mask_on = cfg.MODEL.MASK_ON
        self.ota = cfg.MODEL.OTA
        self.mask_thres = cfg.MODEL.DDETRS.MASK_THRES
        self.new_mask_head = cfg.MODEL.DDETRS.NEW_MASK_HEAD
        self.use_raft = cfg.MODEL.DDETRS.USE_RAFT
        self.use_rel_coord = cfg.MODEL.DDETRS.USE_REL_COORD
        self.num_queries = cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES
        self.mode_free_inference = cfg.MODEL.MODE_FREE_MATCHING_INFERENCE
        self.transform_eval = cfg.MODEL.PANO_TRANSFORM_EVAL
        self.pano_temp = cfg.MODEL.PANO_TEMPERATURE
        self.pano_temp_fg = cfg.MODEL.PANO_TEMPERATURE_CLIP_FG
        self.overlap_threshold = cfg.MODEL.OVERLAP_THRESHOLD

        self.train_labels = get_openseg_labels("coco_panoptic", prompt_engineered=True)


        # Transformer parameters:
        hidden_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
        nheads = cfg.MODEL.DDETRS.NHEADS
        dim_feedforward = cfg.MODEL.DDETRS.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.DDETRS.DEC_LAYERS

        num_feature_levels = cfg.MODEL.DDETRS.NUM_FEATURE_LEVELS
        two_stage = cfg.MODEL.DDETRS.TWO_STAGE
        two_stage_num_proposals = cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS

        # Loss parameters:
        mask_weight = cfg.MODEL.DDETRS.MASK_WEIGHT
        dice_weight = cfg.MODEL.DDETRS.DICE_WEIGHT
        giou_weight = cfg.MODEL.DDETRS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DDETRS.L1_WEIGHT
        class_weight = cfg.MODEL.DDETRS.CLASS_WEIGHT
        deep_supervision = cfg.MODEL.DDETRS.DEEP_SUPERVISION
        focal_alpha = cfg.MODEL.DDETRS.FOCAL_ALPHA
        # Cost parameters (for label assignment):
        set_cost_class = cfg.MODEL.DDETRS.SET_COST_CLASS
        set_cost_bbox = cfg.MODEL.DDETRS.SET_COST_BOX
        set_cost_giou = cfg.MODEL.DDETRS.SET_COST_GIOU
        set_cost_mask = cfg.MODEL.DDETRS.SET_COST_MASK
        set_cost_dice = cfg.MODEL.DDETRS.SET_COST_DICE
        panoptic_box_loss = cfg.MODEL.DDETRS.PANOPTIC_BOX_LOSS
        # Backbone
        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels  # only take [c3 c4 c5] from resnet and gengrate c6 later
        backbone.strides = d2_backbone.feature_strides

        # Transformer & Early Fusion
        if cfg.MODEL.DDETRS.USE_DINO:
            transformer_class = DeformableTransformerVLDINO
        else:
            transformer_class = DeformableTransformerVL
        transformer = transformer_class(
        d_model= hidden_dim,
        nhead=nheads,
        num_encoder_layers=cfg.MODEL.DDETRS.ENC_LAYERS,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=cfg.MODEL.DDETRS.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=cfg.MODEL.DDETRS.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.DDETRS.ENC_N_POINTS,
        two_stage=two_stage,
        two_stage_num_proposals=two_stage_num_proposals,
        use_checkpoint=cfg.MODEL.DDETRS.USE_CHECKPOINT,
        look_forward_twice=cfg.MODEL.DDETRS.LOOK_FORWARD_TWICE,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        cfg=cfg)
        self.object_mask_threshold = cfg.MODEL.OBJECT_MASK_THRESHOLD
        self.use_bg_for_pano = cfg.TEST.USE_BG_FOR_PANO_ON
        self.bg_cls_agnostic = cfg.TEST.BG_CLS_AGNOSTIC
        
        # DETR
        if cfg.MODEL.DDETRS.USE_DINO:
            detr_class = DeformableDETRDINO
        else:
            detr_class = DeformableDETR
        model = detr_class(
        backbone,
        transformer,
        num_queries=self.num_queries,
        num_feature_levels=num_feature_levels,
        aux_loss=deep_supervision,
        with_box_refine=True,
        two_stage=two_stage,
        mixed_selection=cfg.MODEL.DDETRS.MIXED_SELECTION,
        cfg=cfg)

        # Language (text encoder and tokenizer)
        self.parallel_det = cfg.MODEL.PARALLEL_DET
        # Here we use BERT as the text encoder in a hard-code way
        self.tokenizer = AutoTokenizer.from_pretrained("projects/HIPIE/bert-base-uncased")
        if self.parallel_det:
            self.text_encoder = BertEncoder(cfg)
        else:
            self.text_encoder = nn.Sequential(OrderedDict([("body", BertEncoder(cfg))]))
        if cfg.MODEL.FREEZE_TEXT_ENCODER:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        # DETR + Segmentation (CondInst)
        if cfg.MODEL.DDETRS.USE_DINO:
            model_class = DDETRSegmUniDN
        else:
            model_class = DDETRSegmUni
        self.detr = model_class(model, rel_coord=self.use_rel_coord, ota=self.ota, 
        new_mask_head=self.new_mask_head, use_raft=self.use_raft, mask_out_stride=self.mask_stride, 
        decouple_tgt=cfg.MODEL.DECOUPLE_TGT, cls_pool_type=cfg.MODEL.CLS_POOL_TYPE,
        use_iou_branch=cfg.MODEL.USE_IOU_BRANCH, cfg=cfg)

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcherVL(
            cost_class=set_cost_class,
            cost_bbox=set_cost_bbox,
            cost_giou=set_cost_giou,
            cost_mask=set_cost_mask,
            cost_dice=set_cost_dice,
            panoptic_box_loss=panoptic_box_loss)
        

        bg_cost_class =  cfg.MODEL.DDETRS.BG_CLASS_WEIGHT
        bg_cost_mask = cfg.MODEL.DDETRS.BG_MASK_WEIGHT
        bg_cost_dice =  cfg.MODEL.DDETRS.BG_DICE_WEIGHT 
        matcher_bg = HungarianMatcherBG(
           bg_cost_class, 
           bg_cost_mask, 
           bg_cost_dice,
        )

        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight, \
            "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.DDETRS.USE_DINO:
            weight_dict_dn = {"loss_ce_dn": class_weight, "loss_bbox_dn": l1_weight, "loss_giou_dn": giou_weight}
            aux_weight_dict_dn = {}
            for i in range(dec_layers - 1):
                aux_weight_dict_dn.update({k + f"_{i}": v for k, v in weight_dict_dn.items()})
            weight_dict_dn.update(aux_weight_dict_dn)
            weight_dict.update(weight_dict_dn)

        if cfg.MODEL.BOXINST.ENABLED:
            losses = ['labelsVL', 'boxes', 'masks_boxinst']
        else:
            losses = ['labelsVL', 'boxes', 'masks']

        if cfg.MODEL.DDETRS.USE_DINO:
            criterion_class = DINOCriterion
        else:
            criterion_class = SetCriterion
        self.criterion = criterion_class(matcher, weight_dict, losses, focal_alpha=focal_alpha, ota=self.ota, 
        still_cls_for_encoder=cfg.MODEL.STILL_CLS_FOR_ENCODER, cfg=cfg,matcher_bg=matcher_bg,panoptic_box_loss=panoptic_box_loss)
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.use_lsj = cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj"
        
        # BoxInst
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self.num_fg = cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS
        self.num_bg = cfg.MODEL.DDETRS.TWO_STAGE_NUM_BG_PROPOSALS

        # Loss weights for different tasks
        self.loss_weight_det = cfg.SOLVER.LOSS_WEIGHT_DET
        self.loss_weight_grd = cfg.SOLVER.LOSS_WEIGHT_GRD
        if cfg.SAM.ENABLED:
            sam,predictor = build_sam(cfg.SAM.CHECKPOINT,cfg.SAM.TYPE)
            self.sam = sam
            self.sam_predictor = predictor
        else:
            self.sam = None
        self.max_pool_token_test = cfg.TEST.MAX_POOL
        self.enable_clip_train = False
        if cfg.MODEL.CLIP.ENABLED or cfg.MODEL.CLIP.ENABLED_TRAIN:
            self.enable_clip = cfg.MODEL.CLIP.ENABLED
            self.enable_clip_train = cfg.MODEL.CLIP.ENABLED_TRAIN
            if self.enable_clip_train:
                for p in self.text_encoder.parameters():
                    p.requires_grad_(False)
            self.clip = MaskCLIP(name=cfg.MODEL.CLIP.NAME)
            self.clip_fg_a = cfg.MODEL.CLIP.FG_IOU_A 
            self.clip_fg_b = cfg.MODEL.CLIP.FG_IOU_B
            self.clip_alpha = cfg.MODEL.CLIP.ALPHA
            self.clip_agg_mode = cfg.MODEL.CLIP.AGG_MODE
            self.clip_beta = cfg.MODEL.CLIP.BETA
        else:
            self.enable_clip = False
    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """

        # images = self.preprocess_image(batched_inputs)
        # output = self.detr(images)
        task_list = [x["task"] for x in batched_inputs]
        assert len(set(task_list)) == 1
        task = task_list[0]
        if self.training:
            if self.boxinst_enabled:
                images, targets = self.prepare_image_targets_boxinst(batched_inputs)
            else:
                images = self.preprocess_image(batched_inputs)
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances)
            # captions: list[str]
            captions = [x["expressions"] for x in batched_inputs]
            if self.parallel_det:
                language_dict_features = self.forward_text(captions, device="cuda", task=task)
            else:
                language_dict_features = self.forward_text(captions, device="cuda")
            output, loss_dict = self.detr.coco_forward(images, targets, self.criterion, train=True, language_dict_features=language_dict_features, task=task)
            # loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if self.detr.decouple_decoder and '_maskdino' in k: # hack, do not drop mask dino loss
                    continue
                if k in weight_dict:
                    if task == "detection":
                        loss_dict[k] *= (weight_dict[k] * self.loss_weight_det)
                    elif task == "grounding":
                        loss_dict[k] *= (weight_dict[k] * self.loss_weight_grd)
                    else:
                        raise ValueError("task should be detection or grounding")
            return loss_dict
        else:
            torch.cuda.empty_cache()
            images = self.preprocess_image(batched_inputs)
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # targets = self.prepare_targets(gt_instances)
            # captions: list[str]
            captions = [x["expressions"] for x in batched_inputs]
            if task == "grounding":
                positive_map_label_to_token = {1: [0]}
            elif task == "detection":
                positive_map_label_to_token = batched_inputs[0]["positive_map_label_to_token"] # defaultdict(<class 'list'>, {1: [1], 2: [3], 3: [5], 4: [7], 5: [9], 6: [11], 7: [13], 8: [15], 9: [17], 10: [19, 20], 11: [22, 23, 24], 12: [26, 27], 13: [29, 30], 14: [32], 15: [34], 16: [36], 17: [38], 18: [40], 19: [42], 20: [44], 21: [46], 22: [48], 23: [50], 24: [52, 53, 54], 25: [56], 26: [58], 27: [60, 61], 28: [63], 29: [65], 30: [67, 68, 69], 31: [71, 72], 32: [74, 75], 33: [77, 78], 34: [80], 35: [82, 83], 36: [85, 86], 37: [88, 89], 38: [91, 92], 39: [94, 95, 96], 40: [98], 41: [100, 101], 42: [103], 43: [105], 44: [107], 45: [109], 46: [111], 47: [113], 48: [115], 49: [117], 50: [119], 51: [121, 122, 123], 52: [125], 53: [127, 128], 54: [130], 55: [132, 133], 56: [135], 57: [137], 58: [139], 59: [141, 142, 143], 60: [145], 61: [147, 148], 62: [150], 63: [152], 64: [154], 65: [156], 66: [158], 67: [160], 68: [162, 163], 69: [165], 70: [167], 71: [169, 170], 72: [172], 73: [174], 74: [176], 75: [178], 76: [180], 77: [182], 78: [184, 185], 79: [187, 188, 189], 80: [191, 192]})
            else:
                raise ValueError("task must be detection or grounding")
            num_classes = len(positive_map_label_to_token) # num_classes during testing

            if self.parallel_det:
                language_dict_features = self.forward_text(captions, device="cuda", task=task)
            else:
                language_dict_features = self.forward_text(captions, device="cuda")
            is_thing = [x['is_thing'] for x in batched_inputs]
            bg_queries_lang = None
            if self.detr.bg_query_from_lang:
                bg_queries_lang = self.prepare_bg_queries_lan(language_dict_features['hidden'][0],positive_map_label_to_token,is_thing[0])
            output, loss_dict = self.detr.coco_inference(images, None, self.criterion, train=False, language_dict_features=language_dict_features, task=task,
                                                         bg_queries_lang=bg_queries_lang)
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            if self.detr.use_iou_branch:
                iou_pred = output["pred_boxious"]
            else:
                iou_pred = [None]
            # mask_pred = mask_pred[:,:,0]
            #print(self.sam)
            test_labels = list([x.get('open_seg_labels') for x in batched_inputs])
            sizes = list([(x.get("height", image_size[0]),x.get("width", image_size[1]) ) for  x,image_size in zip(batched_inputs,images.image_sizes)])
            denormalized_images = ImageList.from_tensors(
                [x["image"].to(self.device) / 255.0 for x in batched_inputs]
            )
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes, positive_map_label_to_token, num_classes, task=task, iou_pred=iou_pred,is_thing=is_thing,sizes=sizes,output=output,
                                     bg_queries_lang=bg_queries_lang,test_labels=test_labels,images=denormalized_images)
            #print(positive_map_label_to_token)
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = segmentation_postprocess(results_per_image['instances'], height, width)
                    results_per_image['instances'] = r
                    #results_per_image['sem_seg'] =  F.interpolate(results_per_image['sem_seg'].unsqueeze(1), size=(height, width), mode='nearest').squeeze(1)
                    #panoptic_seg,segm_info = results_per_image['panoptic_seg']
                    #panoptic_seg =  F.interpolate(F.one_hot(panoptic_seg.long()).permute(2,0,1).unsqueeze(0).float(), size=(height, width), mode='nearest').argmax(1).squeeze(0)
                    #results_per_image['panoptic_seg'] = (panoptic_seg,segm_info)
                    if self.sam is not None:
                        boxes = r.pred_boxes.tensor.detach().cpu()#.numpy()
                        mask_size = r.pred_masks.shape[1:] # e.g. [100, 333, 500]
                        #print(boxes[:,3].max(),boxes[:,2].max())
                        img_processed = input_per_image['image'].permute(1,2,0).cpu().numpy().astype(np.uint8)
                        h_large,w_large = img_processed.shape[:2]
                        boxes[:,[0,2]] *= 1.0 * h_large / height
                        boxes[:,[1,3]] *= 1.0 * w_large / width
                        boxes = boxes.int()
                        #print(height,width,boxes.shape)
                        #assert boxes.shape[0] == 1
                        
                        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes.to(self.sam_predictor.device), img_processed.shape[:2])
                        self.sam_predictor.set_image(img_processed)
                        masks, _, _ = self.sam_predictor.predict_torch(point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False) # e.g. 100, 1, 800, 1201
                        #print(masks.dtype)
                        
                        masks = F.interpolate(masks.float(),mask_size,mode='area')
                        masks = (masks > 0.5).long().type(torch.uint8).detach().cpu()
                        masks = masks.squeeze(1)
                        r.pred_masks = masks
                        #print(boxes.shape)
                        #print(boxes.max())
                        #print(masks.shape,type(masks),masks.max())
                        #print(r.pred_masks.shape)
                        
                        #print(img_processed.shape)
                    processed_results.append(results_per_image)
                    # # visualization (assume single gpu and batch=1)
                    # caption = captions[0]
                    # ori_images = [cv2.imread(x["file_name"]) for x in batched_inputs][0]
                    # boxes = r.pred_boxes.tensor.cpu().numpy()
                    # boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                    # boxes = boxes.tolist()
                    # masks = r.pred_masks.cpu().numpy().astype(np.float32)
                    # save_images = ori_images.astype(np.float32)
                    # classes = r.pred_classes.cpu().numpy()
                    # # cv2.putText(save_images, caption, (100, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
                    # for (box, mask, class_idx) in zip(boxes, masks, classes):
                    #     color = COCO_CATEGORIES[int(class_idx)]["color"]
                    #     color_mask = np.array(color) * mask[:, :, None] * 0.3
                    #     save_images += color_mask
                    #     x1, y1, w, h = box
                    #     cv2.rectangle(save_images, (int(x1), int(y1)), (int(x1+w), int(y1+h)), tuple(color), thickness=2)
                    # save_images = save_images.clip(0, 255)
                    # save_path = batched_inputs[0]["file_name"].split("/")[-1]
                    # cv2.imwrite(save_path, save_images)
                    # # print(caption)
                return processed_results
            else:
                return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # for language-guided detection, classification loss is computed based on the positive map
            positive_map = targets_per_image.positive_map # (N, 256) or (1, 1). N is number of objects per image
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if hasattr(targets_per_image, "gt_masks"):
                if self.use_lsj:
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                if self.use_amp:
                    gt_masks = gt_masks.half()
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, "image_size": image_size_xyxy, 
                "positive_map": positive_map,'is_thing':targets_per_image.is_thing })
            else:
                new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "image_size": image_size_xyxy, 
                "positive_map": positive_map,'is_thing':targets_per_image.is_thing })
        return new_targets

    def prepare_targets_boxinst(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # for language-guided detection, classification loss is computed based on the positive map
            positive_map = targets_per_image.positive_map # (N, 256) or (1, 1). N is number of objects per image
            if self.use_amp:
                gt_boxes = gt_boxes.half()
                image_size_xyxy = image_size_xyxy.half()
            if self.use_lsj:
                raise NotImplementedError
            else:
                gt_masks = targets_per_image.gt_bitmasks_full
            if self.use_amp:
                gt_masks = gt_masks.half()
            image_color_similarity = targets_per_image.image_color_similarity
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'masks': gt_masks, "image_size": image_size_xyxy, 
            "positive_map": positive_map, "image_color_similarity": image_color_similarity})
        return new_targets

    def panoptic_inference(self, mask_cls, mask_pred,is_thing):
        scores, labels = mask_cls.max(-1)
        mask_pred = mask_pred.sigmoid()
        
        keep = (scores > self.object_mask_threshold) #&  labels.ne(0) 

        # added process
        # if self.transform_eval:
        #     T = self.pano_temp
        #     scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)


        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        #cur_mask_cls = mask_cls[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing =  is_thing.get(int(pred_class+1),True)
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info
        
    @torch.no_grad()
    def inference(self, box_cls, box_pred, mask_pred, image_sizes, positive_map_label_to_token, num_classes, score_thres=0.0, task=None, iou_pred=None,is_thing=None,sizes=None,output=None,
                  bg_queries_lang=None,test_labels=None,images=None):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        if task == "detection":
            max_num_inst = 100
        elif task == "grounding":
            max_num_inst = 1
        else:
            raise ValueError("task must be detection or grounding")
        assert len(box_cls) == len(image_sizes)
        results = []
        bg_end = self.num_bg
        fg_start = self.num_bg
        if self.detr.bg_query_from_lang:
            bg_end = len(bg_queries_lang)
            fg_start = self.num_bg + bg_end
            #breakpoint()
            assert self.num_bg + self.num_fg + len(bg_queries_lang) == box_cls.shape[1]
        box_cls_bg = box_cls[:,:bg_end]
        box_pred_bg = box_pred[:,:bg_end]
        mask_pred_bg = mask_pred[:,:bg_end]

        box_cls = box_cls[:,fg_start:]
        box_pred  = box_pred[:,fg_start:]
        mask_pred  = mask_pred[:,fg_start:]
        iou_pred = iou_pred[:,fg_start:]
        if self.detr.decouple_decoder:
            assert output is not None
            box_cls_bg = output['pred_logits_maskdino']
            box_pred_bg = output['pred_boxes_maskdino']
            mask_pred_bg = output['pred_masks_maskdino'] # F.interpolate(output['pred_masks_maskdino'],scale_factor=2.0,mode='bilinear',align_corners=False)
            mask_pred_bg = mask_pred_bg.unsqueeze(2)

        for i, (logits_per_image, box_pred_per_image, image_size, iou_per_image) in enumerate(zip(
            box_cls, box_pred, image_sizes, iou_pred
        )):
            has_thing =  any(is_thing[i].values())
            if has_thing or 1:
                if self.ota:
                    # NMS
                    logits_per_image = convert_grounding_to_od_logits(logits_per_image.unsqueeze(0), num_classes, positive_map_label_to_token,is_thing=is_thing[i],mode='FG' if has_thing else None,model_free=self.mode_free_inference,
                                                                    max_pool=self.max_pool_token_test)[0]
                    #logits_per_image = logits_per_image[0] # (num_query, C)
                    if self.enable_clip:
                        is_thing_mask = logits_per_image[:1] == -9999.0
                        is_thing_mask = ~is_thing_mask
                        if self.transform_eval and logits_per_image.shape[-1] > 1:
                            logits_per_image_p = F.softmax(logits_per_image.sigmoid() / self.pano_temp_fg, dim=-1)  # already sigmoid
                        else:
                            logits_per_image_p = logits_per_image.sigmoid()
                        prob = self.get_clip_logits(
                            i=i,
                            test_labels=test_labels,
                            mask_pred_results=mask_pred[i][None,:,0],
                            images=images,
                            pred_open_prob=logits_per_image_p,
                            alpha=self.clip_alpha,
                            beta=self.clip_beta,
                        ).sigmoid() * is_thing_mask.float()
                        if iou_per_image is not None:
                            prob = torch.sqrt((prob**self.clip_fg_a) * (iou_per_image.sigmoid() ** self.clip_fg_b) ) # (num_query, C)
                    else:   
                        prob = logits_per_image.sigmoid()
                        # cls x iou
                        if iou_per_image is not None:
                            prob = torch.sqrt(prob * iou_per_image.sigmoid()) # (num_query, C)
                            #breakpoint()
                    # filter low-score predictions
                    if score_thres > 0.0:
                        valid_mask = (prob > score_thres)
                        num_valid = torch.sum(valid_mask).item()
                        num_inst = min(num_valid, max_num_inst)
                        prob[~valid_mask] = -1.0 # set to invalid status
                    else:
                        num_inst = max_num_inst
                    
                    # pre-NMS for duplicate removal
                    nms_scores,idxs = torch.max(prob,1)
                    #nms_scores,_ = iou_per_image.sigmoid().max(1)
                    boxes_before_nms = box_cxcywh_to_xyxy(box_pred_per_image)
                    keep_indices = ops.batched_nms(boxes_before_nms,nms_scores,idxs,0.7)  
                    prob = prob[keep_indices]
                    num_inst = min(num_inst,prob.numel())
                    box_pred_per_image = box_pred_per_image[keep_indices]
                    if mask_pred is not None:
                        mask_pred_i = mask_pred[i][keep_indices]
                    
                    if not self.demo_only:
                        # from the remaining queries (N' x C), picking up topk
                        topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)
                        scores = topk_values
                        topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                        # topk_boxes = topk_indexes // logits_per_image.shape[1]
                        labels = topk_indexes % logits_per_image.shape[1]
                        scores_per_image = scores
                        labels_per_image = labels

                        box_pred_per_image = box_pred_per_image[topk_boxes]
                        if mask_pred is not None:
                            mask_pred_i = mask_pred_i[topk_boxes]
                    else:
                        # Demo Only
                        scores_per_image = nms_scores[keep_indices]
                        labels_per_image = idxs[keep_indices]
                        valid_indices = scores_per_image > score_thres
                        box_pred_per_image = box_pred_per_image[valid_indices]
                        scores_per_image = scores_per_image[valid_indices]
                        labels_per_image = labels_per_image[valid_indices]
                        mask_pred_i = mask_pred_i[valid_indices]
                else:
                    logits_per_image = convert_grounding_to_od_logits(logits_per_image.unsqueeze(0), num_classes, positive_map_label_to_token,is_thing=is_thing[i],mode='FG',model_free=self.mode_free_inference,
                                                                    max_pool=self.max_pool_token_test)
                    logits_per_image = logits_per_image[0] # (num_query, C)
                    prob = logits_per_image.sigmoid()
                    # cls x iou
                    if iou_per_image is not None:
                        prob = torch.sqrt(prob * iou_per_image.sigmoid()) # (num_query, C)
                    # filter low-score predictions
                    if score_thres > 0.0:
                        valid_mask = (prob > score_thres)
                        num_valid = torch.sum(valid_mask).item()
                        num_inst = min(num_valid, max_num_inst)
                        prob[~valid_mask] = -1.0 # set to invalid status
                    else:
                        num_inst = max_num_inst
                    topk_values, topk_indexes = torch.topk(prob.view(-1), num_inst, dim=0)
                    scores = topk_values
                    topk_boxes = torch.div(topk_indexes, logits_per_image.shape[1], rounding_mode='floor')
                    # topk_boxes = topk_indexes // logits_per_image.shape[1]
                    labels = topk_indexes % logits_per_image.shape[1]

                    scores_per_image = scores
                    labels_per_image = labels

                    box_pred_per_image = box_pred_per_image[topk_boxes]
                    if mask_pred is not None:
                        mask_pred_i = mask_pred[i][topk_boxes]
                
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
                result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
                # import pdb;pdb.set_trace()
                if self.mask_on:
                    N, C, H, W = mask_pred_i.shape
                    mask = F.interpolate(mask_pred_i, size=(H*self.mask_stride, W*self.mask_stride), mode='bilinear', align_corners=False)
                    mask = mask.sigmoid() > self.mask_thres
                    # import pdb;pdb.set_trace()
                    mask = mask[:,:,:image_size[0],:image_size[1]]
                    result.pred_masks = mask
                    
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
            else:
                result = Instances(image_size)
                result.pred_boxes = Boxes([])
                result.pred_classes = []
                result.scores = []
            # sem and panoptic
            if task == "detection":
                bg_cls_per_image = box_cls_bg[i]
                bg_mask_per_image = mask_pred_bg[i]
                if self.detr.decouple_decoder and self.detr.mask_dino_fixed_linear_head:
                    logits_per_image_bg = bg_cls_per_image
                    logits_per_image_bg[:80] = -9999
                else:
                    if self.use_bg_for_pano or self.bg_cls_agnostic:
                        logits_per_image_bg = convert_grounding_to_od_logits(bg_cls_per_image.unsqueeze(0), num_classes, positive_map_label_to_token,is_thing=is_thing[i],mode=None,model_free=self.mode_free_inference,
                                                                             max_pool=self.max_pool_token_test)[0]
                    else:
                        logits_per_image_bg = convert_grounding_to_od_logits(bg_cls_per_image.unsqueeze(0), num_classes, positive_map_label_to_token,is_thing=is_thing[i],mode='BG',model_free=self.mode_free_inference,
                                                                             max_pool=self.max_pool_token_test)[0]
                if self.use_bg_for_pano:
                    logits_per_image_all = logits_per_image_bg
                    mask_per_image_all = mask_pred_bg[i]
                else:
                    logits_per_image_all = torch.cat([logits_per_image[keep_indices],logits_per_image_bg],dim=0)
                    mask_per_image_all = torch.cat([mask_pred[i][keep_indices],mask_pred_bg[i]],dim=0)
                N, C, H, W = mask_per_image_all.shape
                valid_cls_mask = logits_per_image_all == -9999.0
                valid_cls_mask = ~valid_cls_mask
                if self.transform_eval:
                    logits_per_image_all = F.softmax(logits_per_image_all.sigmoid() / self.pano_temp, dim=-1)  # already sigmoid
                else:
                    logits_per_image_all = logits_per_image_all.sigmoid()
                mask_per_image_all = F.interpolate(mask_per_image_all, size=(H*self.mask_stride, W*self.mask_stride), mode='bilinear', align_corners=False)
                mask_per_image_all = mask_per_image_all[:,:,:image_size[0],:image_size[1]] # N_Q X 1 X H X W
                if self.enable_clip:
                    
                    clip_logits = self.get_clip_logits(
                        i=i,
                        test_labels=test_labels,
                        mask_pred_results=mask_per_image_all[None,:,0],
                        images=images,
                        pred_open_prob=logits_per_image_all,
                        alpha=self.clip_alpha,
                        beta=self.clip_beta
                    )
                    
                    logits_per_image_all = clip_logits.softmax(-1) #* valid_cls_mask.float()
                    #logits_per_image_all[:len(topk_boxes)] = logits_per_image_all[:len(topk_boxes)]* is_thing_mask
                # if self.use_bg_for_pano:
                #     result = retry_if_cuda_oom(self.instance_inference)(clip_logits.softmax(-1),mask_per_image_all[:,:],is_thing[i],image_size,box_pred_bg[i])
                
                mask_per_image_all_sem_upsample = F.interpolate(mask_per_image_all, size=sizes[i],mode='bilinear', align_corners=False)[:,0]
                semseg = retry_if_cuda_oom(self.semantic_inference)(logits_per_image_all,mask_per_image_all_sem_upsample)
                panoptic_seg, segments_info = retry_if_cuda_oom(self.panoptic_inference)(logits_per_image_all,mask_per_image_all_sem_upsample,is_thing[i])
            else:
                panoptic_seg = None
                segments_info = None
                semseg = None
            results.append(
                dict(
                    instances=result,
                    panoptic_seg=(panoptic_seg, segments_info),
                    sem_seg=semseg
                )
            )
        return results
    
    def instance_inference(self, mask_cls, mask_pred,is_thing,image_size,boxes):
        # mask_pred is already processed to have the same shape as original input
        #image_size = mask_pred.shape[-2:]
        num_queries,num_classes = mask_cls.shape
        # [Q, K]
        scores =  mask_cls
        num_classes = mask_cls.shape[1]
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(100, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, num_classes,rounding_mode='trunc')
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        boxes = boxes[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        keep = torch.zeros_like(scores_per_image).bool()
        for i, lab in enumerate(labels_per_image):
            #keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            keep[i] = is_thing.get(int(lab+1),True)
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]
        boxes = boxes[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        #result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        result.pred_boxes = Boxes(box_cxcywh_to_xyxy(boxes))
        result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid() * result.pred_masks).sum((1,2,3)) / (result.pred_masks.sum((1,2,3))+ 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    
    def get_clip_logits(self,i,test_labels,mask_pred_results,images,
                        pred_open_prob,alpha=0.35,beta=0.7):
        '''
        mask_pred_results: N X Q X H X W
        images: denormalized image ImageList
        
        '''
        test_labels_lst = [x['name'].split(',') for x in test_labels[i]]
        test_labels_lst = list(test_labels_lst)
        train_labels = [x.get('name').split(',') for x in self.train_labels]
        train_labels = {l for label in train_labels for l in label}
        labels = prompt_labels(test_labels_lst, 'photo')
        category_overlapping_list = []
        for test_label in test_labels_lst:
            category_overlapping_list.append(not set(train_labels).isdisjoint(set(test_label)))

        # if self.with_bg and pred_open_logits.shape[-1] == len(self.test_labels) + 1:
        #     category_overlapping_list.append(False)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, device=pred_open_prob.device, dtype=torch.long
        )
        text_embed = self.clip.build_text_embed(labels, verbose=True,always_cache=True).to(pred_open_prob.device)
        #mask_pred_results = mask_per_image_all[None,:,0]
        clip_results = self.clip(
            images.tensor,
            mask_pred_results,
            text_embed,
            labels,
        )
        mask_pred_open_logits = clip_results['mask_pred_open_logits'][0]
        if mask_pred_open_logits.shape[-1] == 1:
            mask_pred_open_prob = mask_pred_open_logits.sigmoid()
        else:
            mask_pred_open_prob = mask_pred_open_logits.softmax(dim=-1)
        if self.clip_agg_mode == 'ADD':
            pred_open_logits_base = (
                    (pred_open_prob * (1 - alpha) + mask_pred_open_prob*alpha + 1e-9).log()
                    # * outputs["logit_scale"]
                    * category_overlapping_mask
                )
            pred_open_logits_novel = (
                (pred_open_prob * (1 - beta) + mask_pred_open_prob * beta + 1e-9).log()
                # * outputs["logit_scale"]
                * (1 - category_overlapping_mask)
            )
        else:
            pred_open_logits_base = (
                    (pred_open_prob ** (1 - alpha) * mask_pred_open_prob**alpha).log()
                    # * outputs["logit_scale"]
                    * category_overlapping_mask
                )
            pred_open_logits_novel = (
                (pred_open_prob ** (1 - beta) * mask_pred_open_prob**beta).log()
                # * outputs["logit_scale"]
                * (1 - category_overlapping_mask)
            )
        pred_open_logits = pred_open_logits_base + pred_open_logits_novel
        return pred_open_logits
    
    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        T = self.pano_temp
        # mask_cls = mask_cls.sigmoid()
        # if self.transform_eval:
        #     mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        if self.use_lsj and self.training:
            image_sizes = [x["instances"].image_size for x in batched_inputs]
            input_masks = [x["padding_mask"].to(self.device) for x in batched_inputs]
            H, W = images[0].size()[-2:]
            images_new = torch.zeros((len(images), 3, H, W), device=self.device)
            for i in range(len(images)):
                h, w = image_sizes[i]
                images_new[i, :, :h, :w] = images[i][:, :h, :w]
            outputs = NestedTensor(images_new, torch.stack(input_masks, dim=0))
            outputs.image_sizes = image_sizes
            return outputs
        else:
            images = ImageList.from_tensors(images)
            return images

    def forward_text(self, captions, device, task=None):
        if self.enable_clip_train:
            return self.forward_text_clip(captions, device, task)
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                                                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)
            sep_token = self.tokenizer('.').input_ids[1]  
            tokenizer_input = {"input_ids": tokenized.input_ids,
                            "attention_mask": tokenized.attention_mask}
            if self.parallel_det:
                language_dict_features = self.text_encoder(tokenizer_input, task=task,sep=sep_token) # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
            else:
                assert len(self.text_encoder) == 1
                language_dict_features = self.text_encoder[0](tokenizer_input,sep=sep_token) # dict with keys: ['aggregate', 'embedded', 'masks', 'hidden']
            # language_dict_features["masks"] is equal to tokenizer_input["attention_mask"]
            # aggregate: (bs, 768), embedded: (bs, L, 768), masks: (bs, 768), hidden: (bs, L, 768) L=256 here
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return language_dict_features

    def forward_text_clip(self, captions, device,task):
        # self.enable_clip_train
        bs = len(captions)
        if task == 'grounding':
            labels = [[x] for x in captions]
            text_embed = self.clip.build_text_embed(labels, verbose=True).to(device) # BS X D_emb
            text_embed = text_embed[:,None]  # BS X 1 X D_emb
            masks = torch.ones(bs,1).float().to(device)
        else:
            labels = [[y for y in x.strip().split(',')] for x in captions[0].split('.')]
            labels = prompt_labels(labels, 'photo')
            text_embed = self.clip.build_text_embed(labels, verbose=True).to(device) # N X D_emb
            n = text_embed.shape[0]
            masks = torch.ones(bs,n).float().to(device)
            text_embed = text_embed[None,].repeat(bs,1,1)
        # hidden , mask
        language_dict_features = dict(
            masks = masks,
            hidden = text_embed
        )
        return language_dict_features

    def prepare_image_targets_boxinst(self, batched_inputs, size_divisibility=32):
        original_images = [x["image"].to(self.device) for x in batched_inputs] # [tensor((3,H,W))] len=bs
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm)

        original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images] # [torch.ones(H, W),...] len=bs

        # mask out the bottom area where the COCO dataset probably has wrong annotations
        for i in range(len(original_image_masks)):
            im_h = batched_inputs[i]["height"]
            pixels_removed = int(
                self.bottom_pixels_removed *
                float(original_images[i].size(1)) / float(im_h)
            )
            if pixels_removed > 0:
                original_image_masks[i][-pixels_removed:, :] = 0

        original_images = ImageList.from_tensors(original_images, size_divisibility)
        original_image_masks = ImageList.from_tensors(
            original_image_masks, size_divisibility, pad_value=0.0
        ) # (bs, H, W) image=1, padding=0
        self.add_bitmasks_from_boxes(
            gt_instances, original_images.tensor, original_image_masks.tensor,
            original_images.tensor.size(-2), original_images.tensor.size(-1)
        )
        
        new_targets = self.prepare_targets_boxinst(gt_instances)
        
        return images_norm, new_targets


    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w):
        stride = self.mask_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )[:, [2, 1, 0]] # RGB-format original images (with padding) (bs, 3, H//4, W//4)
        image_masks = image_masks[:, start::stride, start::stride] # (bs, H//4, W//4)

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy()) # (H, W, 3)
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None] # (1, 3, H//4, W//4)
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],
                self.pairwise_size, self.pairwise_dilation
            ) # (1, 8, H//4, W//4)

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros((im_h, im_w), device=self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0
                per_im_bitmasks_full.append(bitmask_full)

            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0) # (N, H, W)
            per_im_gt_inst.image_color_similarity = torch.cat([
                images_color_similarity for _ in range(len(per_im_gt_inst)) # (N, 8, H//4, W//4)
            ], dim=0)

    def prepare_bg_queries_lan(self,label_enc,positive_map,is_thing_map):
        num_classes = len(positive_map)
        query = torch.zeros(num_classes, label_enc.shape[-1]).to(label_enc.device)
        zero_set = []
        is_thing = torch.zeros(num_classes).bool()
        for label_j in positive_map:
            is_thing[label_j - 1] = is_thing_map.get(label_j,True)
            query[label_j - 1] = label_enc[torch.LongTensor(positive_map[label_j])].mean(0)
        query = self.detr.resizer(query[~is_thing])
        
        return query
def convert_grounding_to_od_logits(logits, num_classes, positive_map, score_agg="MEAN",is_thing={},
                                   mode=None,model_free=False,max_pool=False):
    """
    logits: (bs, num_query, max_seq_len)
    num_classes: 80 for COCO
    """
    assert logits.ndim == 3
    assert positive_map is not None
    if model_free:
        mode = None
    if mode is not None:
        assert mode in ['FG','BG'], f"Invalid mode {mode}"
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_classes).to(logits.device)
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == "MEAN": # True
        for label_j in positive_map:
            if max_pool:
                scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].max(-1)[0]
            else:
                scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].mean(-1)
            if mode == 'FG' and (not is_thing.get(label_j,True)):
                scores[:, :, label_j - 1] = -9999.0
            elif mode == 'BG' and is_thing.get(label_j,True):
                scores[:, :, label_j - 1] = -9999.0
    else:
        raise NotImplementedError
    return scores


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(
        image_masks[None, None], kernel_size=kernel_size,
        dilation=dilation
    )
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights