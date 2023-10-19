import json
import os
import collections

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager
import os.path as osp
import yaml
FILE_DIR = osp.dirname(osp.abspath(__file__))
FILE_DIR = os.path.join(FILE_DIR,'odinw_35.yaml')

with open(FILE_DIR) as f:
    _ODINW_35_LST = yaml.safe_load(f.read())

_CATEGORIES = list(_ODINW_35_LST.keys())

_PREDEFINED_SPLITS_ODINW = {}
for dataset_name,splits in _ODINW_35_LST.items():
    for split_name, split_args in splits.items():
        final_name = 'odinw_{}_{}'.format(dataset_name,split_name)
        img_dir = split_args['img_dir']
        annot_root =  split_args['ann_file']
        name = "odinw_{}".format(dataset_name)
        _PREDEFINED_SPLITS_ODINW[final_name] = (
            split_name,
            img_dir,
            annot_root,
            name
        )

def get_metadata():
    # meta = {"thing_dataset_id_to_contiguous_id": {}}
    meta = {}
    return meta


def load_odinw_json(name, image_root, annot_json, metadata,cat):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    with PathManager.open(annot_json) as f:
        json_info = json.load(f)
        
    # build dictionary for grounding
    grd_dict = collections.defaultdict(list)
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)

    ret = []
    for image in json_info["images"]:
        image_id = int(image["id"])
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]

        if 'train' in name and len(grounding_anno) == 0:
            continue

        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "inst_info": grounding_anno,
                "task":"detection",
                "has_stuff": True,
                "dataset_name":cat,
            }
        )

    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_odinw(
    name, metadata, image_root, annot_json,cat_name):
    DatasetCatalog.register(
        name,
        lambda: load_odinw_json(name, image_root, annot_json, metadata,cat_name),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="odinw",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_odinw(root):
    for (
        prefix,
        (split, folder_name, annot_name,cat_name),
    ) in _PREDEFINED_SPLITS_ODINW.items():
        register_odinw(
            prefix,
            get_metadata(),
            os.path.join(root, folder_name),
            os.path.join(root, annot_name),
            cat_name
        )


# _root = os.getenv("DATASET", "datasets")
# register_all_seginw(_root)