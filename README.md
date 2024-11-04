# Bone Marrow Smear Segmentation with Gradio and Deep Learning Pipeline
PCD 2024

> **Note:** This repository is for educational purposes only, and the source code is not publicly available due to medical confidentiality guidelines. All insights, methods, and outcomes shared here are part of a supervised project at our university laboratory.

## Project Overview
This project focuses on the classification and segmentation of bone marrow cells in smear images, using a deep learning pipeline. Our primary goals were to:
- **Classify Bone Marrow Cells:** Achieve high-accuracy classification of various bone marrow cell types.
- **Segment Bone Marrow Smear Images:** Detect and localize cells in complex smear images.

By combining Convolutional Neural Networks (CNNs) with object detection models, we developed a robust pipeline that achieved over 97% classification accuracy and reliable segmentation results.

## Motivation and Dataset
We used the Bone Marrow Cell Classification dataset from Kaggle, which includes 170,000 images of hematologic disease cell types (e.g., leukemia and lymphomas). Initially, the dataset covered 21 classes, including:
- Abnormal eosinophil
- Artefact
- Basophil
- Blast
- Erythroblast
- Eosinophil
- Faggott cell
- Hairy cell
- Smudge cell
- Immature lymphocyte
- Lymphocyte
- Metamyelocyte
- Monocyte
- Myelocyte
- Band neutrophil
- Segmented neutrophil
- Not identifiable
- Other cell
- Proerythroblast
- Plasma cell
- Promyelocyte

Given the significant class imbalance, we reduced the number of classes to 12 and carefully balanced the dataset to ensure effective training across all relevant categories.

## Methodology

### Phase 1: Classification with Convolutional Neural Networks
- **Initial Model:** We began with a simple CNN for classification, aiming to recognize single cells within images. The model struggled to generalize across 21 initial classes due to the dataset's size and diversity.
  
- **Model Selection and Regularization:** We experimented with VGG and ResNet architectures. However, computational limitations led us to adopt MobileNet as a lightweight yet powerful solution. MobileNet allowed for high accuracy without overwhelming resource demands.

- **Regularization and Data Augmentation:**
  - Employed techniques like dropout, batch normalization, and data augmentation to prevent overfitting.
  - Experimented with optimizers like Adam, which proved most effective for our classification task, enhancing the model's convergence and stability.

- **Class Imbalance Management:** We addressed the imbalance by downsampling dominant classes (e.g., Artefact, Metamyelocyte, Blast), refining the dataset to 12 balanced classes, which significantly improved performance.

### Phase 2: Segmentation with YOLOv8
After achieving satisfactory classification results, we proceeded with object detection and segmentation:
- **Inference with YOLOv8:** We employed YOLOv8 to annotate, detect, and segment bone marrow cells within smear images. The YOLOv8 model used MobileNet’s pretrained weights, leveraging its classification insights to inform cell localization.

- **Pipeline Integration:** The YOLOv8 inference phase was fine-tuned to support accurate detection of individual cells in smear images containing numerous overlapping cells. This step enabled precise segmentation, highlighting individual cell boundaries within complex smear samples.

## Tools and Frameworks
- **Deep Learning Libraries:** Primarily used Keras and TensorFlow for model development and training.
- **Interface:** Gradio was implemented to create an interactive user interface, allowing users to visualize predictions and segmentations.

## Results and Achievements
- **High Classification Accuracy:** We achieved an accuracy of over 97% on the validation set after refining classes and implementing MobileNet.
- **Effective Segmentation:** The YOLOv8-based pipeline allowed accurate cell segmentation within complex smear images, essential for further analysis in medical contexts.

## Educational Outcome & Key Learnings
This project provided deep insights into neural networks, CNN optimization, and the application of object detection for medical image analysis. It also demonstrated the importance of balancing datasets and fine-tuning architectures for resource-constrained environments, making it an invaluable experience for practical machine learning applications in medical research.

## Disclaimer
This project was conducted under the supervision of Mme. Dorra DHOUIB and is intended solely for academic use within our university's laboratory. The source code is confidential and will not be made public in compliance with medical data privacy policies.

## Gradio Interface
Below are images of the user interface developed using Gradio:

### Default Interface
![Default Gradio Interface](path/to/default_interface_image.png)

### Uploaded Image for Annotation/Segmentation
![Uploaded Image for Gradio Annotation](path/to/uploaded_image_interface.png)

*Note: Please replace `path/to/default_interface_image.png` and `path/to/uploaded_image_interface.png` with the actual paths to your images.*
