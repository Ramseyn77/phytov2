"""
model.py
========
Architecture PhytoV2 : Transfer Learning EfficientNetV2B0
  - Phase 1 : Feature extraction (base gelee)
  - Phase 2 : Fine-tuning (degel progressif des 30 dernieres couches)
  - Export  : TFLite INT8 pour Android

Auteur : PhytoV2 Project
"""

import tensorflow as tf
from pathlib import Path

# ---------------------------------------------
# CONSTANTES
# ---------------------------------------------

NUM_CLASSES  = 15
IMAGE_SIZE   = (224, 224)
INPUT_SHAPE  = (224, 224, 3)

# Hyperparametres
PHASE1_LR     = 1e-3
PHASE2_LR     = 1e-5
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 30
UNFREEZE_N    = 30   # Nombre de couches a degeler en Phase 2

# ---------------------------------------------
# CONSTRUCTION DU MODELE
# ---------------------------------------------

def build_model(num_classes: int = NUM_CLASSES, dropout_rate: float = 0.3):
    """
    Construit le modele EfficientNetV2B0 avec une tete de classification
    personnalisee.

    Args:
        num_classes  : Nombre de classes de sortie (defaut : 15).
        dropout_rate : Taux de dropout dans la tete (defaut : 0.3).

    Returns:
        (model, base_model) — le modele complet et la backbone separement
        pour pouvoir gerer le degel en Phase 2.
    """
    # Backbone EfficientNetV2B0 pre-entraine ImageNet
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top          = False,
        weights              = "imagenet",
        input_shape          = INPUT_SHAPE,
        include_preprocessing= True,   # normalisation integree
    )
    base_model.trainable = False  # Gele pour la Phase 1

    # Tete de classification
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="PhytoV2_EfficientNetV2B0")
    return model, base_model


# ---------------------------------------------
# COMPILATION
# ---------------------------------------------

def compile_phase1(model: tf.keras.Model) -> tf.keras.Model:
    """Phase 1 : extraction de features, LR eleve, label smoothing 0.1."""
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics   = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


def compile_phase2(model: tf.keras.Model) -> tf.keras.Model:
    """Phase 2 : fine-tuning, LR faible, label smoothing reduit a 0.05."""
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss      = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics   = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


# ---------------------------------------------
# DEGEL PROGRESSIF (Phase 2)
# ---------------------------------------------

def unfreeze_for_finetuning(
    model      : tf.keras.Model,
    base_model : tf.keras.Model,
    n_layers   : int = UNFREEZE_N,
) -> tf.keras.Model:
    """
    Degele les n_layers dernieres couches de la backbone pour le fine-tuning.
    Toutes les couches precedentes restent gelees pour preserver les features
    bas niveau et accelerer l'entrainement.

    Args:
        model      : Modele complet.
        base_model : Backbone EfficientNetV2B0.
        n_layers   : Nombre de couches a degeler depuis la fin.

    Returns:
        Modele avec couches partiellement degelees.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    frozen    = len(base_model.layers) - trainable
    print(f"  Backbone : {frozen} couches gelees | {trainable} couches degelees")
    return model


# ---------------------------------------------
# CALLBACKS
# ---------------------------------------------

def get_callbacks(checkpoint_dir: str = "checkpoints", phase: int = 1):
    """
    Retourne les callbacks pour une phase d'entrainement :
      - ModelCheckpoint : sauvegarde le meilleur modele (val_accuracy)
      - EarlyStopping   : arret si pas d'amelioration pendant 7 epochs
      - ReduceLROnPlateau : divise le LR par 3 si val_loss stagne
      - TensorBoard      : logs pour visualisation

    Args:
        checkpoint_dir : Dossier de sauvegarde des checkpoints.
        phase          : Numero de la phase (1 ou 2).

    Returns:
        Liste de callbacks Keras.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = str(Path(checkpoint_dir) / f"phase{phase}_best.keras"),
            monitor        = "val_accuracy",
            save_best_only = True,
            mode           = "max",
            verbose        = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_accuracy",
            patience             = 7,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.3,
            patience = 4,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
    ]

    try:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir        = f"logs/phase{phase}",
                histogram_freq = 1,
            )
        )
    except Exception:
        pass

    return callbacks


# ---------------------------------------------
# PIPELINE D'ENTRAINEMENT COMPLET
# ---------------------------------------------

def train(
    train_ds       : tf.data.Dataset,
    val_ds         : tf.data.Dataset,
    checkpoint_dir : str = "checkpoints",
):
    """
    Lance l'entrainement en deux phases :
      Phase 1 — Feature extraction : backbone gelee, on entraine uniquement la tete.
      Phase 2 — Fine-tuning        : on degele les 30 dernieres couches de la backbone.

    Args:
        train_ds       : Dataset d'entrainement (avec CutMix).
        val_ds         : Dataset de validation (sans augmentation).
        checkpoint_dir : Dossier de sauvegarde des checkpoints.

    Returns:
        (model, history_phase1, history_phase2)
    """
    model, base_model = build_model()

    # ── Phase 1 ─────────────────────────────────
    print("\n" + "=" * 55)
    print("  PHASE 1 : Feature Extraction  (base gelee)")
    print("=" * 55)
    model = compile_phase1(model)
    model.summary()

    history_p1 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = PHASE1_EPOCHS,
        callbacks       = get_callbacks(checkpoint_dir, phase=1),
    )

    # ── Phase 2 ─────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  PHASE 2 : Fine-tuning  (degel des {UNFREEZE_N} dernieres couches)")
    print("=" * 55)
    model = unfreeze_for_finetuning(model, base_model, n_layers=UNFREEZE_N)
    model = compile_phase2(model)

    history_p2 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = PHASE2_EPOCHS,
        callbacks       = get_callbacks(checkpoint_dir, phase=2),
    )

    return model, history_p1, history_p2


# ---------------------------------------------
# EXPORT TFLITE INT8 (Android)
# ---------------------------------------------

def make_representative_dataset(dataset: tf.data.Dataset, num_batches: int = 10):
    """
    Cree un generateur de dataset representatif pour la quantification INT8.
    Necessaire pour calibrer les plages de valeurs des activations.

    Args:
        dataset    : Dataset de validation (non shuffle, non augmente).
        num_batches: Nombre de batchs utilises pour la calibration.

    Returns:
        Generateur callable.
    """
    def representative_dataset_gen():
        for images, _ in dataset.take(num_batches):
            for i in range(tf.shape(images)[0]):
                yield [tf.expand_dims(images[i], axis=0)]

    return representative_dataset_gen


def export_tflite(
    model                 : tf.keras.Model,
    output_path           : str   = "phytov2_android.tflite",
    representative_dataset        = None,
) -> str:
    """
    Exporte le modele Keras vers TFLite avec quantification INT8 pour Android.

    Args:
        model                  : Modele Keras entraine.
        output_path            : Chemin du fichier .tflite de sortie.
        representative_dataset : Generateur pour calibration INT8.
                                 Si None, quantification dynamique.

    Returns:
        Chemin absolu du fichier .tflite genere.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_dataset is not None:
        converter.representative_dataset         = representative_dataset
        converter.target_spec.supported_ops      = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type           = tf.uint8
        converter.inference_output_type          = tf.uint8
        print("[INFO] Quantification INT8 complete (avec dataset representatif)")
    else:
        print("[INFO] Quantification dynamique (sans dataset representatif)")

    tflite_model = converter.convert()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)

    size_mb = out.stat().st_size / (1024 ** 2)
    print(f"[DONE] TFLite sauvegarde : {out}  ({size_mb:.2f} MB)")
    return str(out)


# ---------------------------------------------
# POINT D'ENTREE
# ---------------------------------------------

if __name__ == "__main__":
    from preprocessing import build_tf_dataset

    train_ds = build_tf_dataset(
        data_dir     = "dataset_seg/train",
        batch_size   = 32,
        apply_cutmix = True,
        num_classes  = NUM_CLASSES,
    )

    val_ds = build_tf_dataset(
        data_dir     = "dataset_seg/valid",
        batch_size   = 32,
        apply_cutmix = False,
        augment      = False,
        shuffle      = False,
        num_classes  = NUM_CLASSES,
    )

    # Entrainement
    model, history_p1, history_p2 = train(train_ds, val_ds)

    # Sauvegarde Keras
    model.save("phytov2_final.keras")
    print("[DONE] Modele Keras sauvegarde : phytov2_final.keras")

    # Export TFLite INT8
    rep_ds = make_representative_dataset(val_ds, num_batches=10)
    export_tflite(model, "phytov2_android.tflite", representative_dataset=rep_ds)
