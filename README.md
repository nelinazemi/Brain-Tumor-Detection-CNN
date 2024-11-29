# Brain Tumor Classification with CNN

This repository provides a solution for brain tumor classification using a Convolutional Neural Network (CNN) in PyTorch. The dataset contains MRI images of four tumor types: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary Tumor**. The project includes dataset preprocessing, model training, evaluation, and visualization.

---

## Features
- **Dataset Visualization**: Explore data distribution by categories.
- **Custom CNN Architecture**: Optimized for classification tasks.
- **Augmentation & Preprocessing**: Applied transformations for better generalization.
- **Training & Validation**: Tracks accuracy and loss trends.
- **Confusion Matrix**: Visualize model performance.

---

## Requirements

Install the required Python libraries:
```bash
pip install torch torchvision torchmetrics kagglehub matplotlib seaborn pandas scikit-learn tqdm
```

---

## Dataset

The dataset is downloaded from Kaggle using `kagglehub`. Ensure your Kaggle API token is configured correctly.

Download command:
```python
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```

---

## Model Architecture

The custom CNN consists of:
- 4 convolutional blocks with ReLU activations and Batch Normalization.
- MaxPooling for down-sampling.
- Adaptive Average Pooling for compact feature representation.
- Fully connected layer for classification.

---

## Usage

1. Clone this repository and install the dependencies.
2. Configure Kaggle API for dataset download.
3. Train the model using:
   ```python
   python train.py
   ```
4. Evaluate the model and visualize performance.

---

## Visualizations
- **Data Distribution**: Explore category balance with bar plots.
- **Training Metrics**: Visualize loss and accuracy trends across epochs.
- **Confusion Matrix**: Evaluate the classification results.

---

## Future Work
- Add more data augmentation techniques.
- Experiment with transfer learning models like ResNet or EfficientNet.
- Optimize hyperparameters using grid or random search.

---

### Credits
This project uses the dataset by Masoud Nickparvar on Kaggle.
