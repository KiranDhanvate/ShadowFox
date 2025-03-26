# Image Classification System

This project is part of the ShadowFox AI/ML internship and focuses on classifying images using deep learning techniques.

## Project Overview

The image classification system uses Convolutional Neural Networks (CNN) to classify images into predefined categories. This system can be used for various applications such as object recognition, face detection, and scene classification.

## Features

- Image preprocessing and augmentation
- CNN model implementation
- Model training and evaluation
- Real-time prediction capabilities
- Performance visualization

## Project Structure

```
image_classification/
├── src/
│   └── image_classifier.py
├── data/
│   └── images/
└── README.md
```

## Setup Instructions

1. Navigate to the project directory:
```bash
cd image_classification
```

2. Install required dependencies:
```bash
pip install -r ../requirements.txt
```

3. Run the image classification script:
```bash
python src/image_classifier.py
```

## Model Details

The system uses a Convolutional Neural Network with the following features:
- Image preprocessing and normalization
- Data augmentation for better generalization
- Transfer learning capabilities
- Performance evaluation using accuracy and confusion matrix
- Visualization of training results

## Data Requirements

The input data should be organized in the following structure:
```
data/images/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF

## Author

[Kiran Dhanvate](https://github.com/KiranDhanvate) 