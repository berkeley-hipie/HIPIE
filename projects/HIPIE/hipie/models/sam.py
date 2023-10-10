#from segment_anything import sam_model_registry, SamPredictor

def build_sam(
        sam_checkpoint= "sam_vit_h_4b8939.pth",
        model_type= "vit_h",
    ):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    return sam,predictor