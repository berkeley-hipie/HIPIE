import os.path as osp

def get_openseg_labels(dataset, prompt_engineered=False):
    """get the labels in double list format,
    e.g. [[background, bag, bed, ...], ["aeroplane"], ...]
    """

    invalid_name = "invalid_class_id"
    assert dataset in [
        "coco",
        "paco",
        "ade20k_150",
        "ade20k_847",
        "coco_panoptic",
        "pascal_context_59",
        "pascal_context_459",
        "pascal_voc_21",
        "lvis_1203",
        'pascal_parts_pano',
        'pascal_parts_merged',
        "cityscapes_panoptic_parts",
        'obj365v2'
    ] or 'seginw' in dataset or 'odinw' in dataset

    label_path = osp.join(
        osp.dirname(osp.abspath(__file__)),
        "openseg_labels",
        f"{dataset}_with_prompt_eng.txt" if prompt_engineered else f"{dataset}.txt",
    )

    # read text in id:name format
    with open(label_path, "r") as f:
        lines = f.read().splitlines()

    categories = []
    for line in lines:
        id, name = line.split(":",maxsplit=1)
        if name == invalid_name:
            continue
        categories.append({"id": int(id), "name": name})

    return categories