# Segmentation Model README

## Overview

This project focuses on developing a deep learning-based segmentation model for detecting abnormalities in chest X-ray images. The **VinBigData Chest X-ray Abnormalities Detection** dataset, comprising 15 classes of abnormalities, was used for this purpose. The goal is to assist radiologists by automatically detecting these abnormalities and presenting the findings in a structured manner.

Given the complexity of medical imaging, we adopted a thoughtful two-step training approach that allowed the model to first learn general features of X-rays before fine-tuning for specific abnormalities. This process was supported by **5-fold cross-validation**, rigorous monitoring of class-wise IoU, and carefully tuned regularization strategies.

### Key Features of the Pipeline:
1. **Two-step Training Process**: 
   - **Phase 1**: In the first 50 epochs, we focused on letting the model learn general features from the X-ray images. Since there are no pre-trained weights specifically for X-ray data, this step allowed the model to gain a strong foundational understanding.
   - **Phase 2**: After saving the weights from the first phase, we fine-tuned the model for an additional 30 epochs, this time applying heavier regularization to prevent overfitting, especially for the more complex abnormality classes.

2. **5-Fold Cross-Validation**: To ensure robustness and prevent overfitting, we used a **5-fold cross-validation** approach, training the model on different subsets of the data. This allowed the model to generalize well across various splits and ensured that we captured the full spectrum of abnormalities.

3. **Faster R-CNN Model**: The segmentation model is based on **Faster R-CNN with a ResNet50 backbone**, a well-established architecture in object detection tasks. The choice of Faster R-CNN allowed us to balance performance and computational cost effectively.

4. **AdamW Optimizer**: During the fine-tuning phase, we switched to the **AdamW optimizer**. AdamW helps by decoupling the weight decay from the gradient updates, leading to better generalization, especially when fine-tuning after initial training.

5. **Class-wise IoU Monitoring**: In the second phase of training, we closely monitored **class-wise IoU**. This metric ensured that the model was learning to detect all abnormality classes effectively, not just a few. Monitoring class-wise IoU allowed us to identify and address imbalances during training, especially for the more challenging classes like **Pulmonary fibrosis** and **Cardiomegaly**.

---

## Files and Notebooks

### 1. `fastercnn_dataloader.ipynb`
**Description**: This notebook handles data loading and pre-processing for the Faster R-CNN model.

#### Key Features:
- **Why the 'No finding' class was removed**: The overwhelming number of "No finding" cases led to significant class imbalance. Removing this class allowed the model to focus on learning from the actual abnormality classes, which ensured a more balanced and effective training process.

- **Image Preprocessing**:
  - All images were resized to **512x512 pixels** to standardize the input size for the model.
  - **Pixel normalization** was applied to ensure consistency in image intensity, and **histogram equalization** was optionally used to enhance image contrast, making abnormalities more prominent.

- **Bounding Box Normalization**:
  - Bounding boxes were normalized to a range of [0, 1], ensuring consistency across images of varying sizes. The bounding boxes were denormalized only during inference to obtain pixel coordinates for visualizing predictions.

- **Custom Dataset Class**:
  - The `LungsAnnotationDataset` class handles loading the images and annotations (bounding boxes) and ensures that the data is in the correct format for model training.

- **Data Visualization**:
  - We provided a `plot_x_ray` function to visualize bounding boxes and labels, which was crucial for verifying that the annotations were correctly placed on the images.

---

### 2. `fasterCNN1.ipynb`
**Description**: This notebook contains the core model training pipeline for Faster R-CNN, including the two-step training process and the 5-fold cross-validation.

#### Key Features:

- **Why a Two-step Training Process?**:
  - The lack of pre-trained X-ray weights meant that we had to allow the model to learn general features from scratch. In the **first phase** of training, we used minimal regularization, letting the model explore and capture essential features of the X-ray images. This step gave the model a strong foundational understanding of the image data.
  - In the **second phase**, we fine-tuned the model using the weights saved from the first phase. During this fine-tuning, we applied **heavier regularization** (like dropout and weight decay) to prevent overfitting, which was observed in some classes during earlier training. This two-step approach allowed the model to generalize better and focus on the abnormalities more effectively in the second phase.

- **5-fold Cross-Validation**:
  - We used **5-fold cross-validation** to ensure that the model could generalize well across different subsets of the data. Each fold provided an opportunity for the model to learn from different portions of the dataset, preventing overfitting to a single training set and ensuring the model learned robust patterns.

  - **Why K-fold?**: K-fold cross-validation is particularly useful when dealing with medical data, as it ensures that the model learns from diverse samples, including different patient demographics and variations in imaging. By averaging the performance across all folds, we gained a more accurate assessment of the model's capabilities.

- **Phase 1 – Minimal Regularization**:
  - During the first 50 epochs, we applied minimal regularization. This was important to allow the model to explore and learn from the data without being overly constrained by techniques like dropout or heavy weight decay.
  
- **Phase 2 – Fine-tuning with Heavy Regularization**:
  - After observing some overfitting in certain classes (e.g., **Pulmonary fibrosis**), we applied heavier regularization during fine-tuning. **Dropout** was used to reduce overfitting by randomly deactivating neurons during training, while **weight decay** helped in limiting the magnitude of the model weights, thus improving generalization.

  - **AdamW Optimizer**: In the second phase, we switched to the **AdamW optimizer**. AdamW was chosen for its ability to decouple the weight decay from the gradient updates, leading to better generalization during fine-tuning.

  ```python
  optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.0005, betas=(0.9, 0.999), eps=1e-08)
    ```
  - **Why Monitor Class-wise IoU?**: Class-wise IoU was monitored during the second phase to ensure that all classes were being learned equally well. This was particularly important in medical data, where the consequences of poor performance in detecting a specific abnormality could be critical. Monitoring class-wise IoU helped us catch imbalances early on and make necessary adjustments to training, ensuring that no class was neglected.

### 2. `fasterCNN_test.ipynb`
**Description**: This notebook handles the testing and evaluation of the trained model, as well as saving the predictions for further analysis.
**Key Features**:
- **Model Testing and Evaluation**:
  - The fine-tuned Faster R-CNN model was evaluated on a test set, and IoU was again used as the primary metric to measure performance. This phase confirmed whether the model generalized well to unseen data.
- **Saving Predictions**:
  - Predictions were saved to a JSON file, including the bounding boxes and class labels for each test image. Only predictions with confidence scores above a threshold (0.2) were retained, ensuring that only high-confidence detections were kept.
  ```python
  save_predictions_to_json(model, val_data_loader, device, class_brands, filename='results.json')
  ```
- **Class-wise IoU Breakdown**: 
  - We calculated IoU for each class, which allowed us to measure how well the model performed across different abnormality types. This breakdown was crucial for understanding which classes required further improvement and provided insights into where the model could be fine-tuned even further.

## Detailed Approach and Rationale
### Two-step Training Process
We adopted a two-step approach due to the lack of pre-trained weights specific to X-ray images. During the first phase, the goal was to allow the model to develop a general understanding of the X-ray dataset. Without pre-trained weights, starting with minimal regularization was key to letting the model capture core features such as edges, textures, and contrasts in the images.
  - **Why switch to fine-tuning after 50 epochs?** After 50 epochs, the model had already learned most of the general features, but we noticed some overfitting in certain classes during training. At this point, we introduced heavier regularization to avoid memorizing the training data and to improve generalization, especially for more complex classes.

In the second phase, we fine-tuned the model for an additional 30 epochs. We applied heavier dropout and weight decay to prevent overfitting, ensuring the model learned generalized features across the dataset. This strategy helped balance performance across all classes.

## Conclusion
The segmentation model pipeline developed here demonstrates a highly effective, thoughtful approach to detecting abnormalities in chest X-rays. By training the model in two distinct phases and carefully applying regularization strategies, we were able to optimize the model's performance while avoiding overfitting. The combination of Faster R-CNN, K-fold cross-validation, and class-wise IoU monitoring ensured a robust, well-generalized model that can effectively assist radiologists in real-world medical applications.


