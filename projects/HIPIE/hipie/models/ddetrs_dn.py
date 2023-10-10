# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .maskdino.build import build_maskdino
from .maskdino.criterion import SetCriterion as MaskDINOCriterion
from ..util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list, inverse_sigmoid
# from .dcn.deform_conv import DeformConv
from detectron2.structures import Instances
from .conv_with_kaiming_uniform import conv_with_kaiming_uniform
from .deformable_detr.deformable_transformer import agg_lang_feat
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import copy
from .maskdino.matcher import HungarianMatcher as HungarianMatcherMaskDINO
import logging
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_weight_dict(cfg,vl_loss):
    class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
    cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
    cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
    dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
    cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
    deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
    no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT
    mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
    cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
    box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
    cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
    giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
    weight_dict = {"loss_ce": class_weight}
    weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
    weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
    # two stage is the query selection scheme
    if cfg.MODEL.MaskDINO.TWO_STAGE:
        interm_weight_dict = {}
        interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
        weight_dict.update(interm_weight_dict)
    # denoising training
    dn = cfg.MODEL.MaskDINO.DN
    if dn == "standard":
        weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
        dn_losses=["labels","boxes"]
    elif dn == "seg":
        weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
        dn_losses=["labels", "masks","boxes"]
    else:
        dn_losses=[]
    if deep_supervision:
        dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
        aux_weight_dict = {}
        for i in range(dec_layers):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if cfg.MODEL.MaskDINO.BOX_LOSS:
        losses = ["labels", "masks","boxes"]
    else:
        losses = ["labels", "masks"]
    matcher = HungarianMatcherMaskDINO(
        cost_class=cost_class_weight,
        cost_mask=cost_mask_weight,
        cost_dice=cost_dice_weight,
        cost_box=cost_box_weight,
        cost_giou=cost_giou_weight,
        num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        vl_loss = vl_loss,
    )
    return weight_dict,dn_losses,matcher,losses

# Deformable DETR + Segmentaion (CondInst) + De-Noising
class DDETRSegmUniDN(nn.Module):
    def __init__(self, detr, rel_coord=True, ota=False, new_mask_head=False, use_raft=False, mask_out_stride=4, \
        decouple_tgt=False, cls_pool_type="average", use_iou_branch=False, cfg=None):
        super().__init__()
        self.debug_only = False
        self.detr = detr
        self.rel_coord = rel_coord
        self.ota = ota
        self.decouple_tgt = decouple_tgt
        self.cls_pool_type = cls_pool_type
        self.use_iou_branch = use_iou_branch

        self.new_mask_head = new_mask_head
        self.use_raft = use_raft
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        
        self.in_channels = hidden_dim // 32
        self.dynamic_mask_channels = 8
        self.controller_layers = cfg.MODEL.DDETRS.CTRL_LAYERS
        self.max_insts_num = 100
        self.mask_out_stride = mask_out_stride
        self.up_rate = 8 // self.mask_out_stride

        # dynamic_mask_head params
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)

        for contr in self.controller.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)

        self.still_cls_for_encoder = cfg.MODEL.STILL_CLS_FOR_ENCODER

        if new_mask_head:
            self.mask_head = MaskHeadNew(hidden_dim, use_raft=self.use_raft, up_rate=self.up_rate)
        else:
            self.mask_head = MaskHeadSmallConv(hidden_dim, None, hidden_dim, use_raft=self.use_raft, up_rate=self.up_rate)
        
        # denoising
        self.aux_loss = True
        self.num_queries = cfg.MODEL.DDETRS.TWO_STAGE_NUM_PROPOSALS # 900
        self.background_proposals = cfg.MODEL.DDETRS.TWO_STAGE_NUM_BG_PROPOSALS
        self.embed_dim = 256
        self.dynamic_label_enc = cfg.MODEL.DDETRS.DYNAMIC_LABEL_ENC
        if self.dynamic_label_enc:
            self.resizer = FeatureResizer(
                    input_feat_size=768,
                    output_feat_size=self.embed_dim, # 256
                    dropout=cfg.MODEL.DDETRS.DROPOUT
                )
        else:
            self.num_classes = 80 # TODO: This is a hard-code, which needs to be corrected.
            self.label_enc = nn.Embedding(self.num_classes, self.embed_dim) # (80, 256)
        self.dn_number = cfg.MODEL.DDETRS.DN_NUMBER # 100
        self.dp_number = cfg.MODEL.DDETRS.DP_NUMBER # 100
        self.label_noise_ratio = cfg.MODEL.DDETRS.LABEL_NOISE_RATIO # 0.5
        self.box_noise_scale = cfg.MODEL.DDETRS.BOX_NOISE_SCALE # 1.0
        self.bg_query_from_lang =  cfg.MODEL.DDETRS.BG_QUERY_FROM_LANG
        self.bg_weight = cfg.MODEL.DDETRS.FINAL_BG_WEIGHT 
        self.fg_weight = cfg.MODEL.DDETRS.FINAL_FG_WEIGHT
        self.gt_weight = cfg.MODEL.DDETRS.FINAL_GT_WEIGHT    
        self.no_rel_pos = cfg.MODEL.DDETRS.FORCE_NO_LOC
        self.enc_mask = False
        device = torch.device(cfg.MODEL.DEVICE)
        self.decouple_decoder = cfg.MODEL.MASKDINO.ENABLED
        self.background_matcher = cfg.MODEL.DDETRS.BG_MATCHER_MODE
        if self.decouple_decoder:
            self.mask_dino_fixed_linear_head = cfg.MODEL.MASKDINO.FIXED_LINEAR_HEAD
            self.mask_dino_share_encoder = cfg.MODEL.MASKDINO.SHARE_ENCODER
            self.mask_dino_weight = cfg.MODEL.MASKDINO.LOSS_WEIGHT
            self.mask_dino_ckpt =  cfg.MODEL.MASKDINO.PRETRAINED
            output_shape = self.detr.backbone[0].backbone.output_shape()
            mask_dino_cfg = cfg.MODEL.MASKDINO.CONFIG_PATH
            mask_dino_pretrained = cfg.MODEL.MASKDINO.PRETRAINED
            self.mask_dino_share_cls_head = cfg.MODEL.MASKDINO.SHARE_CLS_HEAD
            self.mask_dino,self.mask_dino_cfg= build_maskdino(mask_dino_cfg,output_shape,device=cfg.MODEL.DEVICE,
                                                              num_classes=None if self.mask_dino_fixed_linear_head else cfg.MODEL.DDETRS.HIDDEN_DIM 
                                                              )
            self.mask_dino.to(device)
            cfg = self.mask_dino_cfg
            no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT
            
            dn = cfg.MODEL.MaskDINO.DN
            use_vl_loss = not self.mask_dino_fixed_linear_head
            weight_dict,dn_losses,matcher,losses = get_weight_dict(cfg,vl_loss=use_vl_loss)

            self.mask_dino_criterion = MaskDINOCriterion(100, # Fake num classes
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
                vl_loss=use_vl_loss,
                num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
                oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
                importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
                dn=cfg.MODEL.MaskDINO.DN,
                dn_losses=dn_losses,
                panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
                semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            )
            self.feature_keys = ["res3", "res4", "res5"]
            if self.mask_dino_fixed_linear_head:
                n_mlp = 0
            elif self.mask_dino_share_cls_head:
                n_mlp = 1
            else:
                n_mlp = cfg.MODEL.MaskDINO.DEC_LAYERS+2
            self.mask_dino_cls_embed = _get_clones(self.detr.class_embed[0],n_mlp) 
            
            if self.mask_dino_ckpt:
                # load
                data = torch.load(self.mask_dino_ckpt,map_location=device)
                new_data = {}
                curr_state_dict = self.mask_dino.state_dict()
                
                for k,v in data['model'].items():
                    if k.startswith('sem_seg_head.'):
                        k = k[len('sem_seg_head.'):]
                    if k in curr_state_dict and curr_state_dict[k].shape == v.shape:
                        new_data[k] = v
                #logger = logging.getLogger(__name__)
                res = self.mask_dino.load_state_dict(new_data,strict=False)
                print(f"MASK_DINO has some weights not loaded, typically these are only predication and mask heads:{str(res)}")
            self.mask_dino.to(device)

    def merge_dict(self,dicts,weights):
        final_dict = {}
        for d,w in zip(dicts,weights):
            for k,v in d.items():
                v = v * w
                if k not in final_dict:
                    final_dict[k] = v
                else:
                    final_dict[k] += v
        return final_dict
    
    def post_process_maskdino(self,outputs,language_dict_features,text_masks=None,idx=-1):
        if self.mask_dino_share_cls_head:
            idx = 0
        if self.mask_dino_fixed_linear_head:
            return outputs
        else:
            outputs['pred_logits'] = self.mask_dino_cls_embed[idx](outputs['pred_logits'],language_dict_features)
        if text_masks is not None:
            outputs['text_masks'] = text_masks
        if 'aux_outputs' in outputs:
            aux_outputs = outputs['aux_outputs']
            for i in range(len(aux_outputs)):
                idx = i 
                if self.mask_dino_share_cls_head:
                    idx = 0
                if text_masks is not None:
                    aux_outputs[i]['text_masks'] = text_masks
                aux_outputs[i]['pred_logits'] = self.mask_dino_cls_embed[idx](aux_outputs[i]['pred_logits'],language_dict_features)
        return outputs
    
    def coco_forward(self, samples, gt_targets, criterion, train=False, language_dict_features=None, task=None):
        # if isinstance(samples, (list, torch.Tensor)):
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        if self.debug_only:
            self.debug_data(samples, gt_targets)
        features, pos = self.detr.backbone(samples)

        
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        # split gt tgts
        gt_targets_fg,gt_targets_bg = [],[]
        for gt in gt_targets:
            is_thing = gt['is_thing'] 
            is_thing = is_thing 
            gt_targets_fg.append({
                k:v[is_thing] for k,v in gt.items() if k in ['labels', 'boxes', 'masks', 'positive_map', 'is_thing']
            })
            gt_targets_bg.append({
                k:v[~is_thing] for k,v in gt.items() if k in ['labels', 'boxes', 'masks', 'positive_map', 'is_thing']
            })
            gt_targets_fg[-1]["image_size"]=gt['image_size']
            gt_targets_bg[-1]["image_size"]=gt['image_size']
        for l, feat in enumerate(features):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos[l])
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.num_feature_levels > len(features):
            _len_srcs = len(features)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        # denoising preprocessing
        # prepare label query embedding
        # List (len = bs) each element is {"labels": gt_classes, "boxes": gt_boxes}). gt_boxes is normalized and in cxcywh format
        if self.bg_query_from_lang:
            assert self.dynamic_label_enc
            #lang_feat_bg = language_dict_features["hidden"]
            label_enc_bg = language_dict_features["hidden"] # Bs X L X D
            num_classes = None
        if self.dynamic_label_enc:
            lang_feat_pool = agg_lang_feat(language_dict_features["hidden"], language_dict_features["masks"], pool_type=self.cls_pool_type) # (bs, 768)
            label_enc = self.resizer(lang_feat_pool) # (bs, 256)
            num_classes = None
        else:
            label_enc = self.label_enc
            num_classes = self.num_classes
        no_fg = False

        gt_denoise = gt_targets
        known = [(torch.ones_like(t["labels"])).cuda() for t in gt_targets_fg]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) ==0:
            gt_denoise = gt_targets
            no_fg = True
            known = [len(t["labels"]) for t in gt_denoise]
            assert int(max(known)) > 0
        num_bg_lang = 0
        bg_queries_lang = None
        if self.bg_query_from_lang:
            bg_queries_lang,labels_bg,indices_bg_lang = self.prepare_bg_queries_lan(label_enc_bg,gt_targets)
            num_bg_lang = len(bg_queries_lang)
        input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
            gt_denoise,
            dn_number=self.dn_number, # 100
            label_noise_ratio=self.label_noise_ratio, # 0.5
            box_noise_scale=self.box_noise_scale, # 1.0
            num_queries=self.num_queries+self.background_proposals+num_bg_lang, # 900
            num_classes=num_classes, # 80
            hidden_dim=self.embed_dim, # 256
            label_enc=label_enc, # nn.Embedding(80, 256)
        )



        query_embeds = (input_query_label, input_query_bbox)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, language_dict_features = \
        self.detr.transformer(srcs, masks, poses, query_embeds, mask_on=True, language_dict_features=language_dict_features, task=task, 
        attn_masks=attn_mask,bg_queries_lang=bg_queries_lang,enc_mask=self.enc_mask)
        outputs_mask_enc = None
        if self.enc_mask:
            # features
            enc_outputs_class,output_mem_enc = enc_outputs_class
            dynamic_mask_head_params_enc = self.controller(hs[-1])[:, :, :]
            bs,n_enc,d_emb = dynamic_mask_head_params_enc.shape
            # ref point
            reference_enc = inter_references[-1]
            reference_enc = reference_enc.reshape(1,bs*n_enc,-1)[...,:2] # pts
            num_insts_enc = [n_enc,] * bs
            outputs_mask_enc = self.forward_mask_head_train({}, memory, spatial_shapes, 
                                                                reference_enc, dynamic_mask_head_params_enc, num_insts_enc)
            outputs_mask_enc_filtered = []
            # for (ids,_),msk in zip(indices_bg_lang,outputs_layer_bg_lang['pred_masks']):
            #         outputs_layer_bg_lang_filtered.append(msk[:,ids]) # MASK IS 1 X NUM_Q X NUM_Frame X H X W

  

        if task == "grounding":
            lang_feat_pool = agg_lang_feat(language_dict_features["hidden"], language_dict_features["masks"], pool_type=self.cls_pool_type).unsqueeze(1) # (bs, 1, 768)
        elif task == "detection":
            pass
        else:
            raise ValueError("task must be detection or grounding")
        # memory: [N, \sigma(HiWi), C]
        # hs: [num_encoders, N, num_querries, C]

        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        indices_list = []
        if self.use_iou_branch:
            outputs_ious = []
        enc_lay_num = hs.shape[0]


        outputs_classes_bg = []
        outputs_coords_bg = []
        outputs_masks_bg = []
        indices_list_bg = []

        outputs_classes_gt = []
        outputs_coords_gt = []
        outputs_masks_gt = []
        indices_list_gt = []
        outputs_classes_bg_lang = []
        outputs_coords_bg_lang = []
        outputs_masks_bg_lang = []
        indices_list_bg_lang = []

        gt_indices = self.compute_gt_indices(gt_denoise,dn_meta["dn_num"],dn_meta["dp_num"],dn_meta["single_padding"])
        for lvl in range(enc_lay_num):

            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if task == "grounding":
                outputs_class = self.detr.class_embed[lvl](hs[lvl], lang_feat_pool)
            elif task == "detection":
                outputs_class = self.detr.class_embed[lvl](hs[lvl], language_dict_features["hidden"])
            else:
                raise ValueError("task must be detection or grounding")
            tmp = self.detr.bbox_embed[lvl](hs[lvl])
            if self.use_iou_branch:
                pred_iou = self.detr.iou_head[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            assert (outputs_coord >= 0).all()
            
            # remove dn parts and BG when matching
            padding_size = dn_meta["single_padding"] * (dn_meta["dn_num"]+dn_meta["dp_num"])
            start_fg = padding_size+self.background_proposals+num_bg_lang
            start_bg = padding_size+num_bg_lang
            start_bg_lang = padding_size
            # Fixed BG
            if self.bg_query_from_lang:
                assert  num_bg_lang > 0
                outputs_layer = {
                    'pred_logits': outputs_class[:, start_bg_lang:start_bg, :],
                    'pred_boxes': outputs_coord[:, start_bg_lang:start_bg, :]
                }
                outputs_classes_bg_lang.append(outputs_layer['pred_logits'])
                outputs_coords_bg_lang.append(outputs_layer['pred_boxes'])
                dynamic_mask_head_params_bg_lang = self.controller(hs[lvl])[:,start_bg_lang:start_bg, :]
                reference_bg_lang = reference[:, start_bg_lang:start_bg, :]
                bs,n,d_emb = dynamic_mask_head_params_bg_lang.shape
                dynamic_mask_head_params_bg_lang = dynamic_mask_head_params_bg_lang.reshape(1,bs*n,d_emb)
                num_insts_bg_lang = [num_bg_lang,] * bs
                reference_bg_lang = reference_bg_lang.reshape(1,bs*n,-1)[...,:2] # pts
                outputs_layer_bg_lang = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes, 
                                                            reference_bg_lang, dynamic_mask_head_params_bg_lang, num_insts_bg_lang)
                outputs_layer_bg_lang_filtered = []
                for (ids,_),msk in zip(indices_bg_lang,outputs_layer_bg_lang['pred_masks']):
                    outputs_layer_bg_lang_filtered.append(msk[:,ids]) # MASK IS 1 X NUM_Q X NUM_Frame X H X W
                outputs_masks_bg_lang.append(outputs_layer_bg_lang_filtered)
                indices_list_bg_lang.append(indices_bg_lang)
            # Foreground Filtering
            if dn_meta and dn_meta["single_padding"] > 0:
                padding_size = dn_meta["single_padding"] * (dn_meta["dn_num"]+dn_meta["dp_num"])
                
                outputs_layer = {
                    'pred_logits': outputs_class[:, start_fg:, :],
                    'pred_boxes': outputs_coord[:, start_fg:, :]
                    }
                outputs_classes.append(outputs_layer['pred_logits'])
                outputs_coords.append(outputs_layer['pred_boxes'])
                if self.use_iou_branch:
                    outputs_ious.append(pred_iou[:, start_fg:, :])
                dynamic_mask_head_params = self.controller(hs[lvl])[:, start_fg:, :]    # [bs, num_quries, num_params]
                reference = reference[:, start_fg:, :]
            else:
                raise ValueError("dn_meta should not be None")
                outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
                dynamic_mask_head_params = self.controller(hs[lvl])    # [bs, num_quries, num_params]
            # GT Query PROCESS
            outputs_layer_gt = {
                    'pred_logits': outputs_class[:, :start_bg_lang, :],
                    'pred_boxes': outputs_coord[:, :start_bg_lang, :],
            }
            dynamic_mask_head_params_gt = self.controller(hs[lvl])[:, :start_bg_lang, :]    # [bs, num_quries, num_params]
            reference_gt = reference[:, :start_bg_lang, :]
            bs,n_gt,d_emb = dynamic_mask_head_params_gt.shape
            dynamic_mask_head_params_gt = dynamic_mask_head_params_gt.reshape(1,bs*n_gt,d_emb)
            try:
                reference_gt = reference_gt.reshape(1,bs*n_gt,-1)[...,:2] # pts
            except:
                payload = dict(
                    reference_gt=reference_gt,
                    dynamic_mask_head_params_gt=dynamic_mask_head_params_gt,
                    hs=hs,
                    lvl=lvl,
                    gt_denoise=gt_denoise,
                    gt_targets=gt_targets,
                    reference=reference,
                    dn_meta=dn_meta,
                )
                torch.save(payload,'debug.pth')
                del payload
                raise ValueError
            num_insts_gt = [n_gt,] * bs
            outputs_layer_gt = self.forward_mask_head_train(outputs_layer_gt, memory, spatial_shapes, 
                                                            reference_gt, dynamic_mask_head_params_gt, num_insts_gt)
            outputs_layer_gt_filtered = []
            # gt_indices is prefixed
            for (ids,_),msk in zip(gt_indices,outputs_layer_gt['pred_masks']):
                    outputs_layer_gt_filtered.append(msk[:,ids]) # MASK IS 1 X NUM_Q X NUM_Frame X H X W
                    
            outputs_classes_gt.append(outputs_layer_gt['pred_logits'])
            outputs_coords_gt.append(outputs_layer_gt['pred_boxes'])
            outputs_masks_gt.append(outputs_layer_gt_filtered)
            indices_list_gt.append(gt_indices)
            # BG forward and matching
            if self.background_proposals > 0:
                outputs_layer_bg = {
                        'pred_logits': outputs_class[:, start_bg:start_fg, :],
                        'pred_boxes': outputs_coord[:, start_bg:start_fg, :]
                }
                dynamic_mask_head_params_bg = self.controller(hs[lvl])[:, start_bg:start_fg, :]    # [bs, num_quries, num_params]
                reference_bg = reference[:, start_bg:start_fg:, :]
                bs,n_bg,d_emb = dynamic_mask_head_params_bg.shape
                dynamic_mask_head_params_bg = dynamic_mask_head_params_bg.reshape(1,bs*n_bg,d_emb)
                reference_bg = reference_bg.reshape(1,bs*n_bg,-1)[...,:2] # pts
                num_insts_bg = [n_bg,] * bs
                outputs_layer_bg = self.forward_mask_head_train(outputs_layer_bg, memory, spatial_shapes, 
                                                            reference_bg, dynamic_mask_head_params_bg, num_insts_bg)
                if self.background_matcher == 'MaskDINO':
                    matcher_bg = criterion.matcher
                    indices_bg = matcher_bg.forward(outputs_layer_bg,gt_targets_bg,mask_on=True)
                elif self.background_matcher == 'Mask2Former' :
                    matcher_bg = criterion.matcher_bg
                    indices_bg = matcher_bg.forward(outputs_layer_bg,gt_targets_bg)
                else:
                    raise NotImplemented
                outputs_layer_bg_filtered = []
                for (ids,_),msk in zip(indices_bg,outputs_layer_bg['pred_masks']):
                    outputs_layer_bg_filtered.append(msk[:,ids]) # MASK IS 1 X NUM_Q X NUM_Frame X H X W
                indices_list_bg.append(indices_bg)
                outputs_classes_bg.append(outputs_layer_bg['pred_logits'])
                outputs_coords_bg.append(outputs_layer_bg['pred_boxes'])
                outputs_masks_bg.append(outputs_layer_bg_filtered)
            # for training & log evaluation loss
            # Foreground Matching & NMS
            if self.ota:
                indices, matched_ids = criterion.matcher.forward_ota(outputs_layer, gt_targets_fg)
            else:
                indices = criterion.matcher.forward(outputs_layer, gt_targets_fg)
            indices_list.append(indices)
            reference_points, mask_head_params, num_insts = [], [], []
            for i, indice in enumerate(indices):
                pred_i, tgt_j = indice
                if self.ota:
                    num_insts.append(len(pred_i))
                else:
                    num_insts.append(len(pred_i))
                mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

                # This is the image size after data augmentation (so as the gt boxes & masks)

                orig_h, orig_w = image_sizes[i]
                orig_h = torch.as_tensor(orig_h).to(reference)
                orig_w = torch.as_tensor(orig_w).to(reference)
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                
                ref_cur_f = reference[i].sigmoid()
                ref_cur_f = ref_cur_f[:, :2]
                ref_cur_f = ref_cur_f * scale_f[None, :]
                reference_points.append(ref_cur_f[pred_i].unsqueeze(0))

            # reference_points: [1, \sum{selected_insts}, 2]
            # mask_head_params: [1, \sum{selected_insts}, num_params]
            reference_points = torch.cat(reference_points, dim=1)
            mask_head_params = torch.cat(mask_head_params, dim=1)

            # mask prediction
            has_mask_list = ["masks" in x.keys() for x in gt_targets]
            assert len(set(has_mask_list)) == 1 # must be "all True" or "all False"
            if has_mask_list[0]:
                outputs_layer = self.forward_mask_head_train(outputs_layer, memory, spatial_shapes, 
                                                            reference_points, mask_head_params, num_insts)
            else:
                # avoid unused parameters
                dummy_output = torch.sum(mask_head_params)
                for p in self.mask_head.parameters():
                    dummy_output += 0.0 * p.sum()
                outputs_layer['pred_masks'] = 0.0 * dummy_output
            outputs_masks.append(outputs_layer['pred_masks'])

            # End of FG processing

            # Begin of GT processing
        

        # Post Process BG
        # outputs_class = torch.stack(outputs_classes)
        # outputs_coord = torch.stack(outputs_coords)
        # outputs_mask = outputs_masks
        # if self.use_iou_branch:
        #     outputs_iou = torch.stack(outputs_ious)
        # else:
        #     outputs_iou = None
        # bs, outputs_mask = len(outputs_masks[0]), []
        # outputs_masks: dec_num x bs x [1, num_insts, 1, h, w]

        # denoising postprocessing
        # if dn_meta is not None:
        #     outputs_class, outputs_coord, outputs_iou = self.dn_post_process(
        #         outputs_class, outputs_coord, dn_meta, outputs_iou=outputs_iou
        #     )
        # LANGUAGE 
        if task == "grounding":
            bs, device = language_dict_features["masks"].size(0), language_dict_features["masks"].device
            text_masks = torch.ones((bs, 1), dtype=torch.bool, device=device)
        elif task == "detection":
            text_masks = language_dict_features["masks"]
        else:
            raise ValueError("task must be detection or grounding")

        ###

        if self.decouple_decoder:
            features_maskdino = {k:v.tensors for k,v in zip(self.feature_keys,features)}
            if self.mask_dino_share_encoder:
                encod_feat_l,decod_feat_f,up_masks = self.get_enc_features(memory,spatial_shapes)
                encod_feat_l = list([x.squeeze(2) for x in encod_feat_l])
                pixel_decoder = self.mask_dino.pixel_decoder
                for idx, f in enumerate(pixel_decoder.in_features[:pixel_decoder.num_fpn_levels][::-1]):
                    x = features_maskdino[f].float()
                    lateral_conv = pixel_decoder.lateral_convs[idx]
                    output_conv = pixel_decoder.output_convs[idx]
                    cur_fpn = lateral_conv(x)
                    # Following FPN implementation, we use nearest upsampling here
                    y = cur_fpn + F.interpolate(encod_feat_l[pixel_decoder.high_resolution_index], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
                    y = output_conv(y)
                    encod_feat_l.append(y)
                mask_feat = pixel_decoder.mask_features(encod_feat_l[-1])
                outputs_maskdino,mask_dict = self.mask_dino.predictor(encod_feat_l, mask_feat, None, targets=gt_targets)
            else:
                lang_feat_pool_mask_dino_fwd = agg_lang_feat(language_dict_features["hidden"], language_dict_features["masks"], pool_type=self.cls_pool_type)
                outputs_maskdino,mask_dict = self.mask_dino(features_maskdino,targets=gt_targets,lang_feat_pool=lang_feat_pool_mask_dino_fwd)
            lang = lang_feat_pool if task == 'grounding' else language_dict_features['hidden']
            outputs_maskdino = self.post_process_maskdino(outputs_maskdino,lang,text_masks)
            if mask_dict is not None:
                mask_dict['output_known_lbs_bboxes']  = self.post_process_maskdino(mask_dict['output_known_lbs_bboxes'] ,lang,text_masks)
            outputs_maskdino['interm_outputs'] = self.post_process_maskdino(outputs_maskdino['interm_outputs'] ,lang,text_masks,idx=-2)
            losses_mask_dino = self.mask_dino_criterion(outputs_maskdino, gt_targets,mask_dict)
            for k in list(losses_mask_dino.keys()):
                if k in self.mask_dino_criterion.weight_dict:
                    losses_mask_dino[k] *= self.mask_dino_criterion.weight_dict[k] *  self.mask_dino_weight 
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses_mask_dino.pop(k)
            losses_mask_dino = {k+"_maskdino":v for k,v in losses_mask_dino.items()}
            #breakpoint()
        # outputs['pred_logits'] = outputs_class[-1]
        # outputs['pred_boxes'] = outputs_coord[-1]
        # outputs['pred_masks'] = outputs_mask[-1]
        # if self.use_iou_branch:
        #     outputs['pred_boxious'] = outputs_iou[-1]

        # outputs["text_masks"] = text_masks
        # if self.detr.aux_loss:
        #     if self.use_iou_branch:
        #         outputs['aux_outputs'] = self._set_aux_loss_with_iou(outputs_class, outputs_coord, outputs_mask, outputs_iou)
        #     else:
        #         outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)
        #     for x in outputs['aux_outputs']:
        #         x["text_masks"] = text_masks
        # if self.detr.two_stage and enc_outputs_class is not None:
        #     enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        #     outputs['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, "text_masks": text_masks}

        # add text_masks to dn_meta

        # dn_meta["output_known_lbs_bboxes"]["text_masks"] = text_masks
        # for x in dn_meta["output_known_lbs_bboxes"]['aux_outputs']:
        #     x["text_masks"] = text_masks
        # dn_meta["output_known_lbs_bboxes"]["task"] = task

        # add task
        # outputs["task"] = task
        # # Retrieve the matching between the outputs of the last layer and the targets
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # indices = criterion.matcher(outputs_without_aux, gt_targets)
        out_fg = self.post_processing(outputs_classes,outputs_coords,outputs_masks,text_masks,outputs_ious,None,None,task)
        out_bg = self.post_processing(outputs_classes_bg,outputs_coords_bg,outputs_masks_bg,text_masks,None,None,None,task)
        # FG + BG
        out_gt = self.post_processing(outputs_classes_gt,outputs_coords_gt,outputs_masks_gt,text_masks,None,enc_outputs_class,enc_outputs_coord_unact,task,outputs_mask_enc)
        loss_dict_fg = criterion(out_fg, gt_targets_fg, indices_list, dn_meta)
        loss_dict_bg = criterion(out_bg, gt_targets_bg, indices_list_bg, dn_meta)
        loss_dict_gt = criterion(out_gt, gt_denoise, indices_list_gt, dn_meta)
        out_fg,loss_dict_fg = self.cleanup_losses(out_fg,loss_dict_fg,enc_lay_num)
        out_bg,loss_dict_bg = self.cleanup_losses(out_bg,loss_dict_bg,enc_lay_num)
        out_gt,loss_dict_gt = self.cleanup_losses(out_gt,loss_dict_gt,enc_lay_num)
        all_weights = [self.fg_weight,self.bg_weight,self.gt_weight]
        if len(gt_targets_bg) == 0:
            all_weights[1] = 0.0
        if no_fg:
            all_weights[2] =0.0
        all_losses = [loss_dict_fg,loss_dict_bg,loss_dict_gt]
        if self.bg_query_from_lang:
            out_bg_lang = self.post_processing(outputs_classes_bg_lang,
                                               outputs_coords_bg_lang,
                                               outputs_masks_bg_lang,
                                               text_masks,None,None,None,task)
            loss_dict_bg_lang = criterion(out_bg_lang, gt_targets_bg, indices_list_bg_lang, dn_meta)
            out_bg_lang,loss_dict_bg_lang = self.cleanup_losses(out_bg_lang,loss_dict_bg_lang,enc_lay_num)
            all_losses.append(loss_dict_bg_lang)
            all_weights.append(all_weights[1])
            all_weights[1] = 0.0

        loss_dict = self.merge_dict(all_losses,all_weights)
        # clean up
        if no_fg and self.use_iou_branch:
            loss_iou = torch.stack(outputs_ious).mean() * 0.0 # Hack
            loss_dict['loss_giou'] += loss_iou
        outputs = dict(
            out_fg=out_fg,
            out_bg=out_bg,
            out_gt=out_gt,
        )
        if self.decouple_decoder:
            if task == 'detection':
                loss_dict.update(losses_mask_dino)
                if self.mask_dino_share_encoder:
                    for p in self.mask_dino.pixel_decoder.parameters():
                        loss_dict['loss_giou'] +=  0.0 * p.sum()
            elif task == 'grounding':
                for p in self.mask_dino.parameters():
                    loss_dict['loss_giou'] +=  0.0 * p.sum()
                for p in self.mask_dino_cls_embed.parameters():
                    loss_dict['loss_giou'] +=  0.0 * p.sum()
            else:
                raise NotImplemented
        return outputs,loss_dict

    

    def cleanup_losses(self,outputs, loss_dict,enc_lay_num):
        no_valid_obj = False
        if isinstance(outputs['pred_masks'], list):
            if len(outputs['pred_masks']) == 0:
                no_valid_obj = True
        if no_valid_obj:
            loss_mask = loss_dict["loss_mask"]
            for n, p in self.mask_head.named_parameters():
                loss_mask += 0.0 * torch.sum(p)
            for n, p in self.controller.named_parameters():
                loss_mask += 0.0 * torch.sum(p)
            loss_dict["loss_mask"] = loss_mask
            loss_dict["loss_dice"] = loss_mask
            for i in range(enc_lay_num-1):
                loss_dict["loss_mask_%d"%i] = loss_mask
                loss_dict["loss_dice_%d"%i] = loss_mask
        return outputs, loss_dict
    def post_processing(self,outputs_classes,outputs_coords,outputs_masks,text_masks,outputs_ious=None,
                        enc_outputs_class=None,enc_outputs_coord_unact=None,task=None,outputs_mask_enc=None):
        outputs = {}
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_mask = outputs_masks
        if self.use_iou_branch and outputs_ious is not None:
            outputs_iou = torch.stack(outputs_ious)
        else:
            outputs_iou = None
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        outputs['pred_masks'] = outputs_mask[-1]
        outputs["text_masks"] = text_masks
        if self.detr.aux_loss:
            if self.use_iou_branch and outputs_ious is not None:
                outputs['aux_outputs'] = self._set_aux_loss_with_iou(outputs_class, outputs_coord, outputs_mask, outputs_iou)
            else:
                outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_mask)
            for x in outputs['aux_outputs']:
                x["text_masks"] = text_masks
        if self.detr.two_stage and enc_outputs_class is not None:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            outputs['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord, "text_masks": text_masks}
            if outputs_mask_enc is not None:
                outputs['enc_outputs']['pred_masks'] = outputs_mask_enc['pred_masks']
        outputs["task"] = task
        return outputs

        


    def coco_inference(self, samples, gt_targets, criterion, train=False, language_dict_features=None, 
                       task=None,
                       bg_queries_lang=None):
        image_sizes = samples.image_sizes
        if not isinstance(samples, NestedTensor):
            size_divisibility = getattr(self.detr.backbone[0].backbone, "size_divisibility", 32)
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=size_divisibility)

        features, pos = self.detr.backbone(samples)
        if task == "grounding" or task == "sot":
            lang_feat_pool = agg_lang_feat(language_dict_features["hidden"], language_dict_features["masks"], pool_type=self.cls_pool_type).unsqueeze(1) # (bs, 1, 768)
        elif task == "detection":
            pass
        else:
            raise ValueError("task must be detection or grounding")
        srcs = []
        masks = []
        poses = []
        spatial_shapes = []

        for l, feat in enumerate(features):
            # src: [N, _C, Hi, Wi],
            # mask: [N, Hi, Wi],
            # pos: [N, C, H_p, W_p]
            src, mask = feat.decompose() 
            src_proj_l = self.detr.input_proj[l](src)    # src_proj_l: [N, C, Hi, Wi]
            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos[l])
            n, c, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

        if self.detr.num_feature_levels > len(features):
            _len_srcs = len(features)
            for l in range(_len_srcs, self.detr.num_feature_levels):
                if l == _len_srcs:
                    src = self.detr.input_proj[l](features[-1].tensors)
                else:
                    src = self.detr.input_proj[l](srcs[-1])
                m = masks[0]   # [N, H, W]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)
                n, c, h, w = src.shape
                spatial_shapes.append((h, w))

        input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)
        if getattr(self, "use_deformable_reid_head", False):
            hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, language_dict_features, src_info = \
                self.detr.transformer(srcs, masks, poses, query_embeds, mask_on=True, language_dict_features=language_dict_features, task=task,
                attn_masks=attn_mask, return_src_info=True,bg_queries_lang=bg_queries_lang)
            src_info["reference_points"] = inter_references[-1].detach()
        else:
            hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, language_dict_features = \
                self.detr.transformer(srcs, masks, poses, query_embeds, mask_on=True, language_dict_features=language_dict_features, task=task,
                attn_masks=attn_mask,bg_queries_lang=bg_queries_lang)

        # memory: [N, \sigma(HiWi), C]
        # hs: [num_encoders, N, num_querries, C]
        if self.decouple_decoder:
            features_maskdino = {k:v.tensors for k,v in zip(self.feature_keys,features)}
            if task == "grounding" or task == "sot":
                lang = lang_feat_pool
            else:
                lang = language_dict_features['hidden']
            if self.mask_dino_share_encoder:
                encod_feat_l,decod_feat_f,up_masks = self.get_enc_features(memory,spatial_shapes)
                encod_feat_l = list([x.squeeze(2) for x in encod_feat_l])
                pixel_decoder = self.mask_dino.pixel_decoder
                for idx, f in enumerate(pixel_decoder.in_features[:pixel_decoder.num_fpn_levels][::-1]):
                    x = features_maskdino[f].float()
                    lateral_conv = pixel_decoder.lateral_convs[idx]
                    output_conv = pixel_decoder.output_convs[idx]
                    cur_fpn = lateral_conv(x)
                    # Following FPN implementation, we use nearest upsampling here
                    y = cur_fpn + F.interpolate(encod_feat_l[pixel_decoder.high_resolution_index], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
                    y = output_conv(y)
                    encod_feat_l.append(y)
                mask_feat = pixel_decoder.mask_features(encod_feat_l[-1])
                outputs_maskdino,mask_dict = self.mask_dino.predictor(encod_feat_l, mask_feat, None,)
            else:
                outputs_maskdino,mask_dict = self.mask_dino(features_maskdino)
            
            #outputs_maskdino,mask_dict = self.mask_dino(features_maskdino)
            outputs_maskdino = self.post_process_maskdino(outputs_maskdino,lang,None)
       
        outputs = {}
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        indices_list = []
        if self.use_iou_branch:
            outputs_ious = []

        enc_lay_num = hs.shape[0]
        # for lvl in range(enc_lay_num):
        lvl = enc_lay_num - 1
        if lvl == 0:
            reference = init_reference
        else:
            reference = inter_references[lvl - 1]
        reference = inverse_sigmoid(reference)
        if task == "grounding" or task == "sot":
            outputs_class = self.detr.class_embed[lvl](hs[lvl], lang_feat_pool)
        elif task == "detection":
            outputs_class = self.detr.class_embed[lvl](hs[lvl], language_dict_features["hidden"])
        else:
            raise ValueError("task must be detection or grounding")
        tmp = self.detr.bbox_embed[lvl](hs[lvl])
        if self.use_iou_branch:
            pred_iou = self.detr.iou_head[lvl](hs[lvl])
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        if self.use_iou_branch:
            outputs_ious.append(pred_iou)
            outputs_iou = torch.stack(outputs_ious)
            outputs['pred_boxious'] = outputs_iou[-1]
        # bs, outputs_mask = len(outputs_masks[0]), []
        # outputs_masks: dec_num x bs x [1, num_insts, 1, h, w]
        # import pdb;pdb.set_trace()
        outputs['pred_logits'] = outputs_class[-1]
        outputs['pred_boxes'] = outputs_coord[-1]
        if hasattr(self, "reid_embed_head"):
            if self.use_deformable_reid_head:
                inst_embed = self.reid_embed_head[1](self.reid_embed_head[0](
                    hs[-1], src_info["reference_points"], src_info["src"],
                    src_info["src_spatial_shapes"], src_info["src_level_start_index"], 
                    src_info["src_valid_ratios"], src_info["src_padding_mask"]))
            else:
                inst_embed = self.reid_embed_head(hs[-1])
            outputs['pred_inst_embed'] = inst_embed

        # # Retrieve the matching between the outputs of the last layer and the targets
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # indices = criterion.matcher(outputs_without_aux, gt_targets)
        if train:
            loss_dict = criterion(outputs, gt_targets, indices_list)
        else:
            loss_dict = None
        
        if not train:
            outputs['reference_points'] = inter_references[-2, :, :, :2]
            dynamic_mask_head_params = self.controller(hs[lvl])    # [bs, num_quries, num_params]            
            bs, num_queries, _ = dynamic_mask_head_params.shape
            num_insts = [num_queries for i in range(bs)]
            reference_points = []
            for i, image_size_i in enumerate(image_sizes):
                orig_h, orig_w = image_size_i
                orig_h = torch.as_tensor(orig_h).to(outputs['reference_points'][i])
                orig_w = torch.as_tensor(orig_w).to(outputs['reference_points'][i])
                scale_f = torch.stack([orig_w, orig_h], dim=0)
                ref_cur_f = outputs['reference_points'][i] * scale_f[None, :]
                reference_points.append(ref_cur_f.unsqueeze(0))
            # reference_points: [1, N * num_queries, 2]
            # mask_head_params: [1, N * num_queries, num_params]
            reference_points = torch.cat(reference_points, dim=1)
            mask_head_params = dynamic_mask_head_params.reshape(1, -1, dynamic_mask_head_params.shape[-1])
            # mask prediction
            outputs = self.forward_mask_head_train(outputs, memory, spatial_shapes, 
                                                   reference_points, mask_head_params, num_insts)
            # outputs['pred_masks']: [bs, num_queries, num_frames, H/4, W/4]
            outputs['pred_masks'] = torch.cat(outputs['pred_masks'], dim=0)
            if self.decouple_decoder:
                outputs['pred_masks_maskdino'] = outputs_maskdino['pred_masks']
                outputs['pred_logits_maskdino'] = outputs_maskdino['pred_logits']
                outputs['pred_boxes_maskdino'] = outputs_maskdino['pred_boxes']
        return outputs, loss_dict

    def get_enc_features(self,feats,spatial_shapes):
        bs, _, c = feats.shape
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, spatial_indx: spatial_indx + 1 * h * w, :].reshape(bs, 1, h, w, c).permute(0,4,1,2,3)
            encod_feat_l.append(mem_l)
            spatial_indx += 1 * h * w
        encod_feat_f = []
        for lvl in range(self.detr.num_feature_levels - 1):
            encod_feat_f.append(encod_feat_l[lvl][:, :, 0, :, :]) # [bs, C, hi, wi]
        if self.new_mask_head:
            if self.use_raft:
                decod_feat_f, up_masks = self.mask_head(encod_feat_f)
            else:
                decod_feat_f = self.mask_head(encod_feat_f)
                up_masks = None
        else:
            if self.use_raft:
                decod_feat_f, up_masks = self.mask_head(encod_feat_f, fpns=None)
            else:
                decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
                up_masks = None
        return encod_feat_l,decod_feat_f,up_masks

    def forward_mask_head_train(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, num_insts):
        bs, _, c = feats.shape
        # nq = mask_head_params.shape[1]

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.detr.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, spatial_indx: spatial_indx + 1 * h * w, :].reshape(bs, 1, h, w, c).permute(0,4,1,2,3)
            encod_feat_l.append(mem_l)
            spatial_indx += 1 * h * w
        
        pred_masks = []
        for iframe in range(1):
            encod_feat_f = []
            for lvl in range(self.detr.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :]) # [bs, C, hi, wi]

            # feats = [] # features[3], features[2], features[1]
            # for i in range(self.detr.num_feature_levels - 1, 0, -1):
            #     N, _c, _h, _w = features[i].tensors.shape
            #     feats.append(features[i].tensors.reshape(bs, self.detr.num_frames, _c, _h, _w)[:,iframe,:,:,:])
            
            # decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
            if self.new_mask_head:
                if self.use_raft:
                    decod_feat_f, up_masks = self.mask_head(encod_feat_f)
                else:
                    decod_feat_f = self.mask_head(encod_feat_f)
                    up_masks = None
            else:
                if self.use_raft:
                    decod_feat_f, up_masks = self.mask_head(encod_feat_f, fpns=None)
                else:
                    decod_feat_f = self.mask_head(encod_feat_f, fpns=None)
                    up_masks = None
            # decod_feat_f = self.spatial_decoder(encod_feat_f)[0]  
            # [bs, C/32, H/8, W/8]
            ######### conv ##########
            mask_logits = self.dynamic_mask_with_coords(decod_feat_f, reference_points, mask_head_params, 
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord, up_masks=up_masks)
            # mask_logits: [1, num_queries_all, H/4, W/4]

            # mask_f = mask_logits.unsqueeze(2).reshape(bs, nq, 1, decod_feat_f.shape[-2], decod_feat_f.shape[-1])  # [bs, selected_queries, 1, H/4, W/4]
            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)  
        
        # outputs['pred_masks'] = torch.cat(pred_masks, 2) # [bs, selected_queries, num_frames, H/4, W/4]
        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))
        
        outputs['pred_masks'] = output_pred_masks
        return outputs
    
    def prepare_points(self,batch_idx,dp_number,boxes,labels,pt_noise_scale,single_padding,hidden_dim,
                       batch_size,known_num,label_enc):
        known_bid_pt = batch_idx.repeat(2 * dp_number, 1).view(-1)
        known_pts_boxes = boxes.repeat(2 * dp_number, 1).clone()
        known_pts_labels = labels.repeat(2 * dp_number, 1).view(-1)
        known_pts_bid = batch_idx.repeat(2 * dp_number, 1).view(-1)
        center_offset = torch.rand_like(known_pts_boxes)[...,:2] * 2.0 - 1.0 # [-1,1]
        center_offset = center_offset * pt_noise_scale
        known_pts = known_pts_boxes[...,:2] + center_offset * known_pts_boxes[...,2:]
        known_pts = known_pts.repeat(1,2) # n x 4
        pad_size_pt = int(single_padding * 2 * dp_number)

        input_query_pt_label = torch.zeros(batch_size,pad_size_pt, hidden_dim).cuda() # (N, C)
        input_query_pt_bbox = torch.zeros(batch_size,pad_size_pt, 4).cuda() # (N, 4)

        map_known_indice_pt = torch.tensor([]).to("cuda")
        if self.dynamic_label_enc:
            input_label_embed = label_enc[known_bid_pt] # (n, C)
        else:
            m = known_pts_labels.long().to("cuda")
            input_label_embed = label_enc(m) # (n, C)
        if len(known_num):
            map_known_indice_pt = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice_pt = torch.cat(
                [map_known_indice_pt + single_padding * i for i in range(2 * dp_number)]
            ).long()
        if len(known_bid_pt):
            # (batch idx, known_idx)

            input_query_pt_label[(known_bid_pt.long(), map_known_indice_pt)] = input_label_embed
            input_query_pt_bbox[(known_bid_pt.long(), map_known_indice_pt)] = inverse_sigmoid(known_pts)
        attn_mask = torch.ones(pad_size_pt, pad_size_pt).to("cuda").bool()
        attn_mask[range(pad_size_pt),range(pad_size_pt)] = False

        return input_query_pt_label,input_query_pt_bbox,attn_mask,pad_size_pt
        
    def prepare_bg_queries_lan(self,label_enc,targets):
        label_enc = self.resizer(label_enc)
        label_enc_all = label_enc
        labels = torch.cat([t["labels"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )
        positive_map = torch.cat([t["positive_map"] for t in targets]) # N X L
        isthing = torch.cat([t["is_thing"] for t in targets]) # N X L
        labels_bg,inv = labels[~isthing].unique(return_inverse=True)
        # if len(labels_bg) == 0:
        #     return [],None,None
        d_emb = label_enc.shape[-1]
        label_enc = label_enc[batch_idx][~isthing]
        positive_map = positive_map[~isthing].float()
        label_enc = (label_enc * positive_map.unsqueeze(-1)).sum(1) / (positive_map.squeeze(-1).sum(1,keepdims=True) + 1e-9)
        query_index = []
        for idx,v in enumerate(labels_bg):
            index =torch.where(inv == idx )[0][0].item()
            query_index.append(index)
        indices = []
        bg_queries_lang = label_enc[query_index]
        batch_idx = batch_idx[~isthing]
        for b,v in enumerate(targets):
            src = inv[batch_idx == b]
            tgt = torch.arange(len(src)).to(src.device)
            indices.append((src,tgt))
        
        # append random negatives
        negs = []
        for b,v in enumerate(targets):
            positive_map = v['positive_map']
            label_enc_all_neg = label_enc_all[b][positive_map.sum(0)==0]
            negs.append(label_enc_all_neg)
        label_enc_all_neg = torch.cat(negs,dim=0)
        random_select_negative = torch.randint(0,label_enc_all_neg.shape[1],(20,)).to(label_enc_all_neg)
        label_enc_all_neg = label_enc_all_neg[random_select_negative.long()]
        bg_queries_lang = torch.cat([bg_queries_lang,label_enc_all_neg])
        #label_enc = label_enc[batch_idx][~isthing]
        return bg_queries_lang,labels_bg,indices


        # labels = torch.cat([t["labels"] for t in targets])
        # batch_idx = torch.cat(
        #     [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        # )
        # if self.bg_query_from_lang:
        #     positive_map = torch.cat([t["positive_map"] for t in targets]) # N X L
        #     isthing = torch.cat([t["is_thing"] for t in targets]) # N X L
        #     labels_bg,inv = labels[~isthing].unique(return_inverse=True)
        #     label_enc = self.resizer(label_enc)
        #     d_emb = label_enc.shape[-1]
        #     label_enc = label_enc[batch_idx][~isthing]
        #     positive_map = positive_map[[~isthing]].float()
        #     label_enc = (label_enc * positive_map.unsqueeze(-1)).sum(1) / (positive_map.squeeze(-1).sum(1,keepdims=True) + 1e-9)
        #     query_index = []
        #     for idx,v in enumerate(labels_bg):
        #         index =torch.where(inv == idx )[0][0].item()
        #         query_index.append(index)
        #     indices = []
        #     bg_queries_lang = label_enc[query_index]
        #     batch_idx = batch_idx[~isthing]
        #     for b,v in enumerate(targets):
        #         src = inv[batch_idx == b]
        #         tgt = torch.arange(len(src)).to(src.device)
        #         indices.append((src,tgt))
        # return bg_queries_lang,labels_bg,indices
    def prepare_for_cdn(
        self,
        targets,
        dn_number, # 100
        label_noise_ratio, # 0.5
        box_noise_scale, # 1.0
        num_queries, # 900
        num_classes, # 80
        hidden_dim, # 256
        label_enc,
        dp_number=0,
        pt_noise_scale=0.7
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None,None
        single_padding = int(max(known_num))
        dn_number = dn_number // (int(max(known_num) * 2)) # num of dn-group
        dp_number = dp_number // (int(max(known_num) * 2)) # num of dn-group
        # dn_number is shared in a batch

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
          # if self.bg_query_from_lang
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if not self.dynamic_label_enc:
            if label_noise_ratio > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                    -1
                )  # half of bbox prob
                new_label = torch.randint_like(
                    chosen_indice, 0, num_classes
                )  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
        

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes) # [pos, neg, pos, neg,...]
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2 # (x1, y1)
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2 # (x2, y2)

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2 # (0.5w, 0.5h)
            diff[:, 2:] = known_bboxs[:, 2:] / 2 # (0.5w, 0.5h)

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            ) # -1 or 1
            rand_part = torch.rand_like(known_bboxs) # [0, 1]
            rand_part[negative_idx] += 1.0 # negative [1, 2]
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # transform back to the cxcywh format
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        if self.dynamic_label_enc:
            input_label_embed = label_enc[known_bid] # (n, C)
        else:
            m = known_labels_expaned.long().to("cuda")
            input_label_embed = label_enc(m) # (n, C)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand) # (n, 4)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda() # (N, C)
        padding_bbox = torch.zeros(pad_size, 4).cuda() # (N, 4)

        input_query_label = padding_label.repeat(batch_size, 1, 1) # (bs, N, C)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) # (bs, N, 4)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            # (batch idx, known_idx)
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
        if dp_number > 0:
            input_query_pt_label,input_query_pt_bbox,attn_mask_pt,pad_size_pt = self.prepare_points(
                self,batch_idx,dp_number,boxes,labels,pt_noise_scale,single_padding,hidden_dim,
                       batch_size,known_num)
        else:
            pad_size_pt = 0
        tgt_size = pad_size_pt+pad_size + num_queries 
        """
        For a binary mask, a ``True`` value indicates that the corresponding position is not allowed to attend. 
        For a byte mask, a non-zero value indicates that the corresponding position is not allowed to attend. 
        For a float mask, the mask values will be added to the attention weight.
        """
        attn_mask_full = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask = attn_mask_full[pad_size_pt:,pad_size_pt:]
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True
        attn_mask_full[:pad_size_pt,:] = True
        attn_mask_full[:,:pad_size_pt] = True
        if dp_number > 0:
            attn_mask_full[:pad_size_pt,:pad_size_pt] = attn_mask_pt
        attn_mask = attn_mask_full
        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
            "dp_num":dp_number,
        }
        # TODO: Hanle output
        if dp_number > 0:
            input_query_label = torch.cat([input_query_pt_label,input_query_label],dim=1)
            input_query_bbox = torch.cat([input_query_pt_bbox,input_query_bbox],dim=1)
        return input_query_label, input_query_bbox, attn_mask, dn_meta


    def compute_gt_indices(self,targets,dn_num,dp_num,single_padding):
        dn_idx = []
        # loop over batchsize
        for i in range(len(targets)):
            if len(targets[i]["labels"]) > 0:
                t = torch.arange(0, len(targets[i]["labels"])).long().cuda()
                t = t.unsqueeze(0).repeat(dn_num+dp_num, 1) # shape: (dn_num, n)
                tgt_idx = t.flatten()
                output_idx = (
                    torch.tensor(range(dn_num+dp_num)) * single_padding
                ).long().cuda().unsqueeze(1) + t
                output_idx = output_idx.flatten()
            else:
                output_idx = tgt_idx = torch.tensor([]).long().cuda()

            dn_idx.append((output_idx, tgt_idx))
        return dn_idx

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas, outputs_iou=None):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * (dn_metas["dn_num"]+dn_metas["dp_num"])
            # DN part
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            # matching part
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            if outputs_iou is not None:
                output_known_iou = outputs_iou[:, :, :padding_size, :]
                outputs_iou = outputs_iou[:, :, padding_size:, :]
            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss_dn(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord, outputs_iou



    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


    def dynamic_mask_with_coords(self, mask_feats, reference_points, mask_head_params, num_insts, 
                                 mask_feat_stride, rel_coord=True, up_masks=None):
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        N, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3), 
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]
        
        if rel_coord:
            instance_locations = reference_points
            # instance_locations: [1, num_insts_all, 2]
            # locations: [H*W, 2]
            # import pdb;pdb.set_trace()
            # relative_coords = locations.reshape(1, 1, H, W, 2).repeat(1,num_insts_all,1,1,1)
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            # relative_coords: [1, num_insts_all, H, W, 2]
            # # coords normalization
            # scale = torch.tensor([W, H],device=device)
            # relative_coords = relative_coords.float() / scale[None, None, None, None, :]
            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            # relative_coords: [1, num_insts_all, 2, H*W]
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                if self.no_rel_pos:
                    relative_coords_b *= 0.0
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)
        
        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)
       
        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )

            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs + torch.sum(mask_head_params) * 0.0
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        # mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        if self.use_raft:
            assert up_masks is not None
            inst_idx = 0
            mask_logits_output = []
            for b, n in enumerate(num_insts):
                mask_logits_output.append(self.upsample_preds(mask_logits[inst_idx:inst_idx+n], up_masks[b:b+1]))
                inst_idx += n
            mask_logits = torch.cat(mask_logits_output, dim=0)
        else:
            mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits


    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]

    def _set_aux_loss_dn(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _set_aux_loss_with_iou(self, outputs_class, outputs_coord, outputs_mask, outputs_iou):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c, 'pred_boxious': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1], outputs_iou[:-1])]

    def upsample_preds(self, pred, mask):
        """ Upsample pred [N, 1, H/8, W/8] -> [N, 1, H, W] using convex combination """
        N, _, H, W = pred.shape
        mask = mask.view(1, 1, 9, self.up_rate, self.up_rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_pred = F.unfold(pred, [3,3], padding=1)
        up_pred = up_pred.view(N, 1, 9, 1, 1, H, W)

        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, 1, self.up_rate*H, self.up_rate*W)

    def debug_data(self, samples, gt_targets):
        import numpy as np
        import copy
        import cv2
        import torch.distributed as dist
        import sys
        import time
        mean = np.array([123.675, 116.280, 103.530])
        std = np.array([58.395, 57.120, 57.375])
        default_color = (255,255,255)
        color_list = [x["color"] for x in COCO_CATEGORIES]
        num_color = len(color_list)
        for i in range(len(gt_targets)):
            image = samples.tensors[i].permute((1, 2, 0)).cpu().numpy() * std + mean # (H, W, 3)
            input_mask = samples.mask[i].float().cpu().numpy() * 255 # (H, W)
            image = np.ascontiguousarray(image[:, :, ::-1]).clip(0, 255)
            target = gt_targets[i]
            boxes = target["boxes"].cpu().numpy()
            num_inst = boxes.shape[0]
            for j in range(num_inst):
                cx, cy, w, h = boxes[j] * target["image_size"].cpu().numpy() # image size without padding
                x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
                if "masks" in target:
                    mask = target["masks"][j].cpu().float().numpy() # (H, W)
                    if mask.shape != image.shape[:-1]:
                        ori_h, ori_w = mask.shape
                        mask_new = np.zeros((image.shape[:-1]))
                        mask_new[:ori_h, :ori_w] = mask
                    else:
                        mask_new = mask
                    image[:, :, -1] += 128 * mask_new
                if "inst_id" in target and target["inst_id"][j] != -1:
                    color = color_list[target["inst_id"][j] % num_color]
                else:
                    color = default_color
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
            cv2.imwrite("rank_%02d_batch_%d_img.jpg"%(dist.get_rank(), i), image)
            cv2.imwrite("rank_%02d_batch_%d_mask.jpg"%(dist.get_rank(), i), input_mask)
        time.sleep(5)
        sys.exit(0)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, use_raft=False, up_rate=4):
        super().__init__()
        self.use_raft = use_raft
        if use_raft:
            self.out_stride = up_rate
        else:
            self.out_stride = 2 # original value is 8 (compared with 4x downsampled mask, here should be 2)
        self.up_rate = up_rate
        # inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        inter_dims = [dim, context_dim, context_dim, context_dim, context_dim, context_dim]

        # used after upsampling to reduce dimention of fused features!
        self.lay1 = torch.nn.Conv2d(dim, dim//4, 3, padding=1)
        # self.gn1 = torch.nn.GroupNorm(8, dim//4)
        self.lay2 = torch.nn.Conv2d(dim//4, dim//32, 3, padding=1)
        # self.gn2 = torch.nn.GroupNorm(8, dim//32)

        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        # self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        # self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        # self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        # self.conv_offset = torch.nn.Conv2d(inter_dims[3], 18, 1)#, bias=False)
        # self.dcn = DeformConv(inter_dims[3],inter_dims[4], 3, padding=1)
        self.jia_dcn = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.dim = dim

        if fpn_dims != None:
            self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
            self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for name, m in self.named_modules():
            if name == "conv_offset":
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)
        if self.use_raft:
            self.up_mask_layer = nn.Sequential(
                nn.Conv2d(context_dim, context_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(context_dim, self.up_rate*self.up_rate*9, 1, padding=0))

    def forward(self, x, fpns):
        # enc_p3, enc_p4, enc_p5 = x
        # x = self.lay1(x)
        # x = self.gn1(x)
        # x = F.relu(x)
        # x = self.lay2(x)
        # x = self.gn2(x)
        # x = F.relu(x)

        if fpns != None:
            cur_fpn = self.adapter1(fpns[0])
            if cur_fpn.size(0) != x[-1].size(0):
                cur_fpn = _expand(cur_fpn, x[-1].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-1]) / 2
        else:
            fused_x = x[-1]
        fused_x = self.lay3(fused_x)
        # fused_x = self.gn3(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter2(fpns[1])
            if cur_fpn.size(0) != x[-2].size(0):
                cur_fpn = _expand(cur_fpn, x[-2].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-2]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-2] + F.interpolate(fused_x, size=x[-2].shape[-2:], mode="nearest")
        fused_x = self.lay4(fused_x)
        # fused_x = self.gn4(fused_x)
        fused_x = F.relu(fused_x)

        if fpns != None:
            cur_fpn = self.adapter3(fpns[2])
            if cur_fpn.size(0) != x[-3].size(0):
                cur_fpn = _expand(cur_fpn, x[-3].size(0) // cur_fpn.size(0))
            fused_x = (cur_fpn + x[-3]) / 2 + F.interpolate(fused_x, size=cur_fpn.shape[-2:], mode="nearest")
        else:
            fused_x = x[-3] + F.interpolate(fused_x, size=x[-3].shape[-2:], mode="nearest")
        # dcn for the last layer
        # offset = self.conv_offset(x)
        # x = self.dcn(x,offset)
        fused_x = self.jia_dcn(fused_x)
        # fused_x = self.gn5(fused_x)
        fused_x_fpn = F.relu(fused_x)

        fused_x = self.lay1(fused_x_fpn)
        # fused_x = self.gn1(fused_x)
        fused_x = F.relu(fused_x)
        fused_x = self.lay2(fused_x)
        # fused_x = self.gn2(fused_x)
        fused_x = F.relu(fused_x)

        if self.use_raft:
            up_masks = self.up_mask_layer(fused_x_fpn) # weights used for upsampling the coarse mask predictions
            return fused_x, up_masks
        else:
            return fused_x

class MaskHeadNew(nn.Module):
    """
    22.04.04 New mask head (as same as CondInst)
    """

    def __init__(self, in_channels, channels=128, num_convs=4, sem_loss_on=False, num_classes=80, use_raft=False, up_rate=4):
        super().__init__()

        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        self.num_outputs = 8
        self.use_raft = use_raft
        if use_raft:
            self.out_stride = up_rate
        else:
            self.out_stride = 2 # original value is 8 (compared with 4x downsampled mask, here should be 2)
        self.up_rate = up_rate
        self.sem_loss_on = sem_loss_on

        self.refine = nn.ModuleList()
        for _ in range(3):
            self.refine.append(conv_block(in_channels, channels, 3))

        tower = nn.ModuleList()
        for _ in range(num_convs):
            tower.append(conv_block(channels, channels, 3))
        tower.append(nn.Conv2d(channels, max(self.num_outputs, 1), 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.use_raft:
            self.up_mask_layer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, self.up_rate*self.up_rate*9, 1, padding=0))

        if self.sem_loss_on:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0

            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1)
            )

            self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)


    def forward(self, features):
        """gt_bitmasks_full: (bs, M, H, W), gt_classes: (bs, M)"""
        # NOTE: gt_bitmasks_full has been downsampled by 4 (to reduce latency)
        # Here CondInst uses multiple-level features (p3, p4, p5)
        # -3, -2, -1 corresponds to P3, P4, P5
        for i, f in enumerate([-3, -2, -1]):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p

        mask_feats = self.tower(x)


        if self.use_raft:
            up_masks = self.up_mask_layer(x) # weights used for upsampling the coarse mask predictions
            return mask_feats, up_masks
        else:
            return mask_feats

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def segmentation_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
    ):

    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        # import pdb;pdb.set_trace()
        mask = F.interpolate(results.pred_masks.float(), size=(output_height, output_width), mode='nearest')
        # import pdb;pdb.set_trace()
        mask = mask.squeeze(1).byte()
        results.pred_masks = mask

        # import pdb;pdb.set_trace()
        #  results.pred_masks [N, output-height, output-width]


    return results

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout):
        super().__init__()
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        x = self.layer_norm(x)
        output = self.dropout(x)
        return output
