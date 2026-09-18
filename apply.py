import sys 
from preprocessing import (
    # Segmentation
    get_yolo_model,
    segment_leaf,
    preprocess_dataset,

    # CutMix
    cutmix_batch,

    # Pipeline tf.data
    build_tf_dataset,

    # Constantes
    IMAGE_SIZE,
    YOLO_WEIGHTS,
    YOLO_CONF,
)

# 1. Segmenter le dataset complet
preprocess_dataset(
    input_root  = r"d:\DeepProject\PhytoV2\dataset",
    output_root = r"d:\DeepProject\PhytoV2\dataset_seg",
    use_yolo    = True,
)

# 2. Construire le pipeline d'entrainement
train_ds = build_tf_dataset(
    data_dir    = r"d:\DeepProject\PhytoV2\dataset_seg\train",
    batch_size  = 32,
    apply_cutmix= True,
    num_classes = 15,
)

val_ds = build_tf_dataset(
    data_dir    = r"d:\DeepProject\PhytoV2\dataset_seg\valid",
    batch_size  = 32,
    apply_cutmix= False,   # pas de CutMix sur la validation
    augment     = False,
    shuffle     = False,
    num_classes = 15,
)
