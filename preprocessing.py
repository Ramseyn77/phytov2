"""
preprocessing.py
================
Pipeline de prétraitement pour PhytoV2 :
  1. Segmentation de feuille via YOLOv8-seg (fallback : GrabCut)
  2. Augmentation CutMix pour TensorFlow

Auteur : Sufyane Ramseyn
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------
# CONSTANTES
# ---------------------------------------------

IMAGE_SIZE   = (224, 224)   # Taille cible pour le classifieur
BG_COLOR     = (0, 0, 0)    # Couleur du fond apres masquage (noir)
YOLO_CONF    = 0.25         # Seuil de confiance YOLO
YOLO_WEIGHTS = "yolov8n-seg.pt"  # Telecharge automatiquement si absent

# ---------------------------------------------
# CHARGEMENT DU MODELE YOLO (singleton)
# ---------------------------------------------

_yolo_model = None

def get_yolo_model(weights: str = YOLO_WEIGHTS) -> YOLO:
    """Charge le modele YOLOv8-seg une seule fois (singleton)."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(weights)
    return _yolo_model


# ---------------------------------------------
# SEGMENTATION PAR YOLO
# ---------------------------------------------

def _get_largest_centered_mask(results, img_h: int, img_w: int):
    """
    Parmi tous les masques YOLO detectes, retourne celui dont le centroide
    est le plus proche du centre de l'image (strategie robuste pour les feuilles
    qui ne sont pas une classe COCO explicite).

    Returns:
        masque binaire uint8 (H, W) ou None si aucun masque.
    """
    if results[0].masks is None:
        return None

    cx_img, cy_img = img_w / 2, img_h / 2
    best_mask  = None
    best_score = float("inf")

    for mask_tensor in results[0].masks.data:
        # Masque binaire redimensionne a la taille de l'image originale
        mask = mask_tensor.cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Centroide du masque
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        dist = np.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
        if dist < best_score:
            best_score = dist
            best_mask  = mask

    return best_mask


def _grabcut_fallback(image: np.ndarray) -> np.ndarray:
    """
    Segmentation de secours via GrabCut OpenCV lorsque YOLO
    ne detecte aucun objet.

    Returns:
        masque binaire uint8 (H, W)
    """
    h, w = image.shape[:2]
    mask      = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Rectangle interieur a 10% des bords (suppose contenir la feuille)
    margin_x = int(w * 0.10)
    margin_y = int(h * 0.10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    return binary_mask


def segment_leaf(image: np.ndarray, use_yolo: bool = True) -> np.ndarray:
    """
    Segmente la feuille dans l'image et retourne une image recadree
    sur la bounding box de la feuille avec le fond mis a BG_COLOR,
    redimensionnee a IMAGE_SIZE.

    Args:
        image    : Image BGR lue par OpenCV (H, W, 3).
        use_yolo : Si True, tente d'abord YOLOv8-seg.

    Returns:
        Image segmentee (224, 224, 3) dtype uint8.
    """
    img_h, img_w = image.shape[:2]
    binary_mask  = None

    # Etape 1 : Tentative YOLO
    if use_yolo:
        model   = get_yolo_model()
        results = model(image, conf=YOLO_CONF, verbose=False)
        binary_mask = _get_largest_centered_mask(results, img_h, img_w)

    # Etape 2 : Fallback GrabCut
    if binary_mask is None:
        binary_mask = _grabcut_fallback(image)

    # Etape 3 : Application du masque
    masked_image = image.copy()
    masked_image[binary_mask == 0] = BG_COLOR  # fond -> noir

    # Etape 4 : Crop sur la bounding box de la feuille
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + bw + pad)
        y2 = min(img_h, y + bh + pad)
        masked_image = masked_image[y1:y2, x1:x2]

    # Etape 5 : Resize vers IMAGE_SIZE
    output = cv2.resize(masked_image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    return output


# ---------------------------------------------
# PREPROCESSING DU DATASET COMPLET
# ---------------------------------------------

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_dataset(
    input_root : str,
    output_root: str,
    use_yolo   : bool = True,
    verbose    : bool = True,
) -> None:
    """
    Parcourt toute l'arborescence input_root (train/valid/test / classes / images),
    applique segment_leaf() sur chaque image et sauvegarde le resultat dans
    output_root en conservant la meme structure de dossiers.

    Args:
        input_root  : Racine du dataset original  (ex: "dataset/")
        output_root : Racine du dataset segmente  (ex: "dataset_seg/")
        use_yolo    : Activer la segmentation YOLO.
        verbose     : Affiche la progression.
    """
    input_root  = Path(input_root)
    output_root = Path(output_root)

    image_paths = [
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ]

    total = len(image_paths)
    if verbose:
        print(f"[INFO] {total} images trouvees dans '{input_root}'")

    for idx, img_path in enumerate(image_paths, 1):
        relative = img_path.relative_to(input_root)
        out_path = output_root / relative
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(img_path))
        if image is None:
            if verbose:
                print(f"  [WARN] Impossible de lire : {img_path}")
            continue

        try:
            segmented = segment_leaf(image, use_yolo=use_yolo)
            cv2.imwrite(str(out_path), segmented)
        except Exception as e:
            if verbose:
                print(f"  [ERR] {img_path.name} -> {e}")
            continue

        if verbose and idx % 500 == 0:
            print(f"  [{idx}/{total}] traites...")

    if verbose:
        print(f"[DONE] Dataset segmente sauvegarde dans '{output_root}'")


# ---------------------------------------------
# CUTMIX � AUGMENTATION TENSORFLOW
# ---------------------------------------------

def _cutmix_sample_box(lam: float, img_h: int, img_w: int):
    """
    Calcule les coordonnees d'une boite de decoupe CutMix.
    La surface de la boite � (1 - lam) x surface totale.

    Returns:
        (x1, y1, x2, y2) entiers
    """
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(img_h * cut_ratio)
    cut_w = int(img_w * cut_ratio)

    cx = np.random.randint(img_w)
    cy = np.random.randint(img_h)

    x1 = np.clip(cx - cut_w // 2, 0, img_w)
    y1 = np.clip(cy - cut_h // 2, 0, img_h)
    x2 = np.clip(cx + cut_w // 2, 0, img_w)
    y2 = np.clip(cy + cut_h // 2, 0, img_h)

    return x1, y1, x2, y2


def cutmix_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha : float = 1.0,
):
    """
    Applique CutMix sur un batch d'images TensorFlow.

    Args:
        images : Tensor shape (B, H, W, C), valeurs dans [0, 1].
        labels : Tensor shape (B, num_classes) � one-hot encode.
        alpha  : Parametre de la distribution Beta pour lambda.

    Returns:
        (mixed_images, mixed_labels) � meme shape que les entrees.
    """
    batch_size = tf.shape(images)[0]
    img_h      = IMAGE_SIZE[0]
    img_w      = IMAGE_SIZE[1]

    lam = np.random.beta(alpha, alpha)

    indices = tf.random.shuffle(tf.range(batch_size))
    images2 = tf.gather(images, indices)
    labels2 = tf.gather(labels, indices)

    x1, y1, x2, y2 = _cutmix_sample_box(lam, img_h, img_w)

    lam_real = 1.0 - ((x2 - x1) * (y2 - y1)) / (img_h * img_w)

    # Masque binaire : 0 dans la boite, 1 partout ailleurs
    mask = np.ones((1, img_h, img_w, 1), dtype=np.float32)
    mask[0, y1:y2, x1:x2, 0] = 0.0
    mask_tf = tf.constant(mask)

    mixed_images = images * mask_tf + images2 * (1.0 - mask_tf)
    mixed_labels = lam_real * labels + (1.0 - lam_real) * labels2

    return mixed_images, mixed_labels


# ---------------------------------------------
# CONSTRUCTION DU PIPELINE tf.data
# ---------------------------------------------

def build_tf_dataset(
    data_dir     : str,
    batch_size   : int   = 32,
    apply_cutmix : bool  = True,
    cutmix_alpha : float = 1.0,
    shuffle      : bool  = True,
    num_classes  : int   = 15,
    augment      : bool  = True,
) -> tf.data.Dataset:
    """
    Construit un tf.data.Dataset a partir d'un dossier structure par classe
    (ex: dataset_seg/train/).

    Args:
        data_dir     : Chemin vers le dossier (ex: "dataset_seg/train").
        batch_size   : Taille du batch.
        apply_cutmix : Appliquer CutMix sur les batchs.
        cutmix_alpha : Parametre Beta pour CutMix.
        shuffle      : Melanger les donnees.
        num_classes  : Nombre de classes (pour le one-hot).
        augment      : Appliquer des augmentations standard.

    Returns:
        tf.data.Dataset yielding (images, labels).
    """
    # Chargement depuis arborescence de dossiers
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        image_size  = IMAGE_SIZE,
        batch_size  = None,
        label_mode  = "int",
        shuffle     = shuffle,
    )

    # Normalisation + one-hot
    def to_onehot(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return image, label

    ds = raw_ds.map(to_onehot, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentations standard
    if augment:
        augmentation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ])

        def apply_augment(image, label):
            image = augmentation_layer(image, training=True)
            return image, label

        ds = ds.map(apply_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    ds = ds.batch(batch_size)

    # CutMix
    if apply_cutmix:
        def apply_cutmix_wrapper(images, labels):
            imgs, lbls = tf.py_function(
                func=lambda i, l: cutmix_batch(i, l, cutmix_alpha),
                inp=[images, labels],
                Tout=[tf.float32, tf.float32],
            )
            imgs.set_shape([None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
            lbls.set_shape([None, num_classes])
            return imgs, lbls

        ds = ds.map(apply_cutmix_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------
# POINT D'ENTREE � SEGMENTATION DU DATASET
# ---------------------------------------------

if __name__ == "__main__":
    preprocess_dataset(
        input_root  = "dataset",
        output_root = "dataset_seg",
        use_yolo    = True,
        verbose     = True,
    )
