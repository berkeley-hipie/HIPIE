# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CITYSCAPE_PARTS_CATEGORIES = [
 {'id': 1, 'name': 'ground ', 'color': [81, 13, 96]},
 {'id': 2, 'name': 'road ', 'color': [36, 163, 198]},
 {'id': 3, 'name': 'sidewalk ', 'color': [97, 231, 124]},
 {'id': 4, 'name': 'parking ', 'color': [137, 161, 201]},
 {'id': 5, 'name': 'rail track ', 'color': [180, 21, 219]},
 {'id': 6, 'name': 'building ', 'color': [177, 10, 73]},
 {'id': 7, 'name': 'wall ', 'color': [50, 49, 254]},
 {'id': 8, 'name': 'fence ', 'color': [126, 96, 5]},
 {'id': 9, 'name': 'guard rail ', 'color': [123, 207, 31]},
 {'id': 10, 'name': 'bridge ', 'color': [251, 77, 74]},
 {'id': 11, 'name': 'tunnel ', 'color': [67, 192, 203]},
 {'id': 12, 'name': 'pole ', 'color': [248, 5, 32]},
 {'id': 13, 'name': 'polegroup ', 'color': [20, 179, 65]},
 {'id': 14, 'name': 'traffic light ', 'color': [41, 34, 177]},
 {'id': 15, 'name': 'traffic sign ', 'color': [69, 31, 187]},
 {'id': 16, 'name': 'vegetation ', 'color': [199, 143, 12]},
 {'id': 17, 'name': 'terrain ', 'color': [1, 90, 154]},
 {'id': 18, 'name': 'sky ', 'color': [35, 59, 74]},
 {'id': 19, 'name': 'person torso', 'color': [27, 82, 255]},
 {'id': 20, 'name': 'person head', 'color': [34, 11, 225]},
 {'id': 21, 'name': 'person arm', 'color': [46, 244, 156]},
 {'id': 22, 'name': 'person leg', 'color': [232, 82, 82]},
 {'id': 23, 'name': 'rider torso', 'color': [158, 45, 49]},
 {'id': 24, 'name': 'rider head', 'color': [240, 103, 0]},
 {'id': 25, 'name': 'rider arm', 'color': [24, 133, 4]},
 {'id': 26, 'name': 'rider leg', 'color': [236, 227, 150]},
 {'id': 27, 'name': 'car body', 'color': [137, 54, 156]},
 {'id': 28, 'name': 'car window', 'color': [207, 218, 26]},
 {'id': 29, 'name': 'car wheel', 'color': [101, 204, 242]},
 {'id': 30, 'name': 'car light', 'color': [105, 34, 49]},
 {'id': 31, 'name': 'car license plate', 'color': [169, 204, 235]},
 {'id': 32, 'name': 'car chassis', 'color': [146, 231, 135]},
 {'id': 33, 'name': 'truck body', 'color': [39, 221, 164]},
 {'id': 34, 'name': 'truck window', 'color': [226, 71, 215]},
 {'id': 35, 'name': 'truck wheel', 'color': [67, 116, 70]},
 {'id': 36, 'name': 'truck light', 'color': [100, 224, 95]},
 {'id': 37, 'name': 'truck license plate', 'color': [145, 89, 121]},
 {'id': 38, 'name': 'truck chassis', 'color': [92, 52, 223]},
 {'id': 39, 'name': 'bus body', 'color': [102, 160, 25]},
 {'id': 40, 'name': 'bus window', 'color': [209, 158, 64]},
 {'id': 41, 'name': 'bus wheel', 'color': [36, 204, 176]},
 {'id': 42, 'name': 'bus light', 'color': [7, 97, 96]},
 {'id': 43, 'name': 'bus license plate', 'color': [136, 105, 168]},
 {'id': 44, 'name': 'bus chassis', 'color': [149, 94, 115]},
 {'id': 45, 'name': 'caravan ', 'color': [219, 216, 0]},
 {'id': 46, 'name': 'trailer ', 'color': [98, 42, 130]},
 {'id': 47, 'name': 'train ', 'color': [25, 164, 153]},
 {'id': 48, 'name': 'motorcycle ', 'color': [150, 185, 50]},
 {'id': 49, 'name': 'bicycle ', 'color': [164, 195, 137]},
 {'id': 50, 'name': 'license plate ', 'color': [203, 46, 26]}
]

def _get_parts_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in CITYSCAPE_PARTS_CATEGORIES]
    #assert len(stuff_ids) == 459, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in CITYSCAPE_PARTS_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret



CITYSCAPES_LABEL_GROUP ={2: 1,
 3: 2,
 6: 3,
 7: 4,
 8: 5,
 12: 6,
 14: 7,
 16: 8,
 17: 9,
 18: 10,
 19: 11,
 20: 12,
 21: 13,
 22: 14,
 23: 15,
 24: 16,
 25: 17,
 26: 18,
 27: 19,
 28: 20,
 29: 21,
 30: 22,
 31: 23,
 32: 24,
 33: 25,
 34: 26,
 35: 27,
 36: 28,
 37: 29,
 38: 30,
 39: 31,
 40: 32,
 41: 33,
 42: 34,
 43: 35,
 44: 36,
 47: 37,
 48: 38,
 49: 39,
 50: 40}
def register_all_cityscapes_parts(root):
    data_root = root
    root = os.path.join(root, "cityscapes")
    meta = _get_parts_meta()
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, "images_all", dirname)
        gt_dir = os.path.join(root, "gtFinePanopticPartsRemapped", dirname)
        name = f"cityscapes_panoptic_parts_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="_gtFinePanopticParts.tif", image_ext="_leftImg8bit.png",dataset_name='cityscapes_panoptic_parts')
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            thing_dataset_id_to_contiguous_id={},  # to make Mask2Former happy
            stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            label_group=CITYSCAPES_LABEL_GROUP,
            ignore_label=0,  # NOTE: gt is saved in 16-bit TIFF images
        )

# register_all_ctx59(os.getenv("DETECTRON2_DATASETS", "datasets"))
# register_all_pascal21(os.getenv("DETECTRON2_DATASETS", "datasets"))
# register_all_ctx459(os.getenv("DETECTRON2_DATASETS", "datasets"))
