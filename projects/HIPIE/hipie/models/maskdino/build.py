from detectron2.config import get_cfg
#from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from .config import add_maskdino_config
from .meta_arch.maskdino_head import MaskDINOHead
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.projects.deeplab import add_deeplab_config
#hack
def build_maskdino(config_file,output_shape,device=None,num_classes=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    if device is not None:
        cfg.MODEL.DEVICE = device
    if num_classes is not None:
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    #cfg.freeze()
    decoder = MaskDINOHead(cfg,output_shape)
    return decoder,cfg