# Chest X-ray Pre-processing

## Overview

This project focuses on the pre-processing of chest X-ray images from the [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) Kaggle challenge. The dataset, originally in DICOM format, was converted to PNG and resized to 512x512, significantly reducing its size. Additionally, bounding boxes were normalized to account for the resizing, ensuring the annotations remain accurate.

The pre-processing pipeline consists of two main stages:
1. **Conversion from DICOM to PNG**.
2. **Image resizing, bounding box normalization, and visualization**.

---

## Files

### 1. `convert_dicom_png.ipynb`
**Description**: This notebook handles the conversion of DICOM images to PNG format and normalizes pixel values.

#### Key Steps:
- **DICOM to PNG conversion**: Each DICOM file is loaded, its pixel values normalized, and saved as a PNG image.
- **Input**: DICOM files from the dataset.
- **Output**: PNG files saved in the specified output folder.

#### Example Workflow:
- Specify the input folder containing DICOM files and the output folder where PNG files will be saved.
- The notebook will iterate through all DICOM files, convert them to PNG, and save them.

---

### 2. `Dataset-pre-processing.ipynb`
**Description**: This notebook handles the following:
1. **Image Resizing**: All PNG images are resized to 512x512 dimensions using OpenCV to optimize for computational efficiency.
2. **Bounding Box Normalization**: Bounding boxes are normalized based on the resized image dimensions to ensure their accuracy.
3. **Bounding Box Visualization**: Bounding boxes are drawn on the images to visually confirm correct placement and size after normalization.

#### Key Steps:
- **Resizing**: Each image is resized to 512x512 pixels and saved to a specified output folder.
- **Bounding Box Normalization**: Bounding box coordinates are normalized to a scale of [0, 1] based on the image dimensions, ensuring compatibility with resized images.
- **Visualization**: Bounding boxes and class labels are drawn on the resized images to verify their correctness.

#### Example Workflow:
- Load the bounding box data from the CSV file.
- Extract and append image dimensions (width, height) to the CSV.
- Normalize bounding box coordinates.
- Visualize the bounding boxes on the resized images for confirmation.

---

## How to Run Pre-processing

### 1. **Convert DICOM to PNG**
- Open the `convert_dicom_png.ipynb` notebook.
- Specify the input folder containing DICOM files and the output folder for PNG files.
- Run the notebook to convert all DICOM files to PNG format.

### 2. **Resize Images & Normalize Bounding Boxes**
- Open the `Dataset-pre-processing.ipynb` notebook.
- Specify the input folder containing the PNG images and the CSV file with bounding box data.
- Run the notebook to:
  1. Resize the images to 512x512 pixels.
  2. Normalize the bounding boxes based on the new image dimensions.
  3. (Optional) Visualize the bounding boxes on the resized images.

---

## Dataset Details

- **Original Dataset**: [VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
- **Original Image Format**: DICOM
- **Converted Image Format**: PNG
- **Original Dataset Size**: 260GB (compressed)
- **Resized Dataset Size**: ~70GB (512x512 PNG images)

### Image Resizing:
The images were resized to 512x512 to balance image quality and computational efficiency. Resizing helps to significantly reduce the dataset size while maintaining necessary detail for model training.

### Bounding Box Normalization:
Bounding box coordinates are normalized relative to the image dimensions to maintain accurate annotation. This is crucial for ensuring that bounding boxes remain properly scaled and aligned after resizing.

---

## How to Visualize Bounding Boxes

The `Dataset-pre-processing.ipynb` notebook includes a function to visualize bounding boxes:

- **Function**: `visualize_bounding_boxes(image_id, boxes, labels)`
- This function overlays bounding boxes on the images and displays them to verify the bounding box placement and scaling.

To use this function:
- Provide the `image_id`, bounding box coordinates (`boxes`), and class labels (`labels`).
- The function will display the image with bounding boxes drawn over it.

---

## Conclusion

This pre-processing pipeline ensures that the `VinBigData Chest X-ray` dataset is prepared efficiently for further analysis and model training. The conversion, resizing, and normalization steps are critical for maintaining the quality and accuracy of both the images and their annotations.
