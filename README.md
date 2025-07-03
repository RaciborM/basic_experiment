# Face recognition - basic experiment
# ORL Face Recognition with CNN (Keras/TensorFlow)

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to recognize human faces from the **ORL Face Dataset**. It includes preprocessing, model training, evaluation, and performance visualization.

## Dataset

The ORL face dataset consists of grayscale images of 40 individuals, with 10 images per person. Each image is 92x112 pixels, and contains variations in facial expressions and facial details.

- **Dataset Path** (expected format):
```
ORL_dataset/
├── s1/
│ ├── 1.pgm
│ ├── 2.pgm
│ └── ...
├── s2/
└── ...
```

## Features

- Automatic dataset loading with custom 6/4 train/test split.
- CNN architecture with batch normalization and dropout.
- Data shuffling and batch generation.
- Plotting training and test accuracy/loss.
- Saves accuracy/loss chart as `accuracy_loss_plot.png`.

## Results

