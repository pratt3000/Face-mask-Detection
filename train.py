import argparse
from imutils import paths
from cv2 import cv2
import os
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Flatten,
    Input,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# argument parser
print("----- SETTING PARSER")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, help="name of data folder", required=False, default="dataset"
)
args = vars(parser.parse_args())

# read data
print("----- READING IMAGES FROM DATABASE")
imgs = []
labels = []
img_paths = list(paths.list_images(args["dataset"]))
print(args)
for img_path in img_paths:  # 2478 images

    label = img_path.split(os.path.sep)[-2]
    labels.append(label)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    imgs.append(img)
    # img = img.flatten()

print("----- REFORMATTING DATA VARIABLES")
Y = np.array(labels)
X = np.array(imgs, dtype="float32")

bi = preprocessing.LabelBinarizer()
bi = bi.fit(Y)
Y = bi.transform(Y)
Y = tf.keras.utils.to_categorical(Y)

print("----- CREATING TRAIN TEST SETS")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=2, stratify=Y
)

print("----- DATA AUGMENTATION")
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

print("----- DEFINING MODEL STRUCTURE")
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
)

top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Flatten()(top_model)
top_model = Dense(128, activation="relu")(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(2, activation="softmax")(top_model)

model = Model(inputs=base_model.input, outputs=top_model)

print("----- SETTING TRAIN ARCHITECTURE")
for layer in base_model.layers:
    layer.trainable = False

opt = Adam(lr=0.0004, decay=0.00002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(
    aug.flow(X_train, Y_train, batch_size=32),
    steps_per_epoch=len(X_train)
    // 32,  # how many time data generation is to be done, skip if pc can handle big datasets
    validation_data=(X_test, Y_test),
    validation_steps=len(X_test) // 32,  # same as steps per epoch for validation
    epochs=8,
)

print("----- SAVING MODEL DATA")
model.save("mask_detector.model", save_format="h5")
