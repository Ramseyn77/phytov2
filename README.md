---
annotations_creators:
- expert-generated
language_creators:
- found
language:
- en
license: cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
tags:
- computer-vision
- agriculture
- plant-disease
- leaf-classification
- smart-farming
- precision-agriculture
pretty_name: Plant Leaf Disease Dataset (15 Classes - Pepper, Potato, Tomato)
---

# 🌿 Plant Leaf Disease Dataset (15 Classes - Unsegmented RAW)

## 📌 Dataset Summary
This dataset contains high-quality RGB images of healthy and diseased plant leaves across **3 major agricultural crops**: **Pepper (Bell)**, **Potato**, and **Tomato**. The dataset is structured into **15 distinct classes** covering viral, bacterial, fungal diseases, pest infestations, and healthy controls. 

Unlike segmented datasets, this collection consists of **raw (unsegmented) RGB images**, providing a realistic environment for benchmarking computer vision models, leaf detection, background noise robustness, and edge AI deployment in real-world precision farming scenarios.

---

## 📂 Dataset Structure
The dataset is pre-partitioned into standard subsets:
- **`train/`**: Training images categorized by class folders.
- **`valid/`**: Validation images for hyperparameter tuning.
- **`test/`**: Test images for final model evaluation.

### 🏷️ Class Taxonomy (15 Classes)

| Crop | Class Name | Condition Type |
| :--- | :--- | :--- |
| **Pepper (Bell)** | `Pepper__bell___Bacterial_spot` | Bacterial Disease |
| **Pepper (Bell)** | `Pepper__bell___healthy` | Healthy |
| **Potato** | `Potato___Early_blight` | Fungal Disease |
| **Potato** | `Potato___Late_blight` | Oomycete Disease |
| **Potato** | `Potato___healthy` | Healthy |
| **Tomato** | `Tomato_Bacterial_spot` | Bacterial Disease |
| **Tomato** | `Tomato_Early_blight` | Fungal Disease |
| **Tomato** | `Tomato_Late_blight` | Oomycete Disease |
| **Tomato** | `Tomato_Leaf_Mold` | Fungal Disease |
| **Tomato** | `Tomato_Septoria_leaf_spot` | Fungal Disease |
| **Tomato** | `Tomato_Spider_mites_Two_spotted_spider_mite` | Pest Infestation |
| **Tomato** | `Tomato__Target_Spot` | Fungal Disease |
| **Tomato** | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Viral Disease |
| **Tomato** | `Tomato__Tomato_mosaic_virus` | Viral Disease |
| **Tomato** | `Tomato_healthy` | Healthy |

---

## 🎯 Intended Use Cases
- **Computer Vision & Image Classification**: Benchmark architectures such as EfficientNet, ResNet, Vision Transformers (ViT), and MobileNet.
- **Precision Agriculture**: Train models for automated crop disease diagnosis and early warning systems.
- **Edge AI & Mobile Deployment**: Develop lightweight models (TFLite, ONNX) for real-time mobile and drone applications.
- **Preprocessing & Segmentation Experiments**: Test segmentation pipelines (YOLOv8-seg, GrabCut, SAM) to evaluate performance before and after leaf extraction.

---

## 📜 Citation & License
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Usage**: Free to use for research, academic, and commercial application development with proper attribution.
