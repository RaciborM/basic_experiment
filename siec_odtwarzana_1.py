import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Flatten, Dense, Activation
)
from tensorflow.keras.optimizers import Adam

# seed config
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# main params
IMG_HEIGHT, IMG_WIDTH = 32, 32
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)
NUM_CLASSES = 40
DATASET_PATH = r"your_PATH_to_dataset"
BATCH_SIZE = 64

# data load (6train/4test)
def load_orl_dataset_split(path, target_size=(32, 32)):
    X_train, y_train = [], []
    X_test, y_test = [], []
    label = 0
    for person_dir in sorted(os.listdir(path)):
        person_path = os.path.join(path, person_dir)
        if os.path.isdir(person_path):
            images = sorted(os.listdir(person_path), key=lambda x: int(os.path.splitext(x)[0]))
            for i, img_name in enumerate(images):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, target_size)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)
                if i < 6:
                    X_train.append(img)
                    y_train.append(label)
                else:
                    X_test.append(img)
                    y_test.append(label)
            label += 1
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_orl_dataset_split(DATASET_PATH)

def batch_generator(X, y, batch_size):
    num_samples = X.shape[0]
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        for offset in range(0, num_samples, batch_size):
            batch_X = X[offset:offset+batch_size]
            batch_y = y[offset:offset+batch_size]
            yield batch_X, batch_y

train_generator = batch_generator(X_train, y_train, BATCH_SIZE)

# model
def build_face_recognition_cnn(input_shape, num_classes):
    model = Sequential()

    # conv1
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # conv2
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # fully-conn(dense)
    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # output
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

model = build_face_recognition_cnn(INPUT_SHAPE, NUM_CLASSES)

model.compile(
    optimizer=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=20,
    steps_per_epoch=400,
    validation_data=(X_test, y_test)
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nDokładność na zbiorze testowym: {test_acc * 100:.2f}%")

# plots
plt.figure(figsize=(12, 5))

# acc
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Dokładność treningowa i testowa')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

# loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Strata treningowa i testowa')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_loss_plot.png")
plt.show()
