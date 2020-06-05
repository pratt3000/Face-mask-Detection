# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument(
    "-p",
    "--plot",
    type=str,
    default="plot.png",
    help="path to output loss/accuracy plot",
)
ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="mask_detector.model",
    help="path to output face mask detector model",
)
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,

INIT_LR = 1e-4  # reduce when iterations increase
EPOCHS = 8  # iters...ations
BS = 32  # batch size

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("****** loading images ******")
imagePaths = list(paths.list_images(args["dataset"]))  # addresses
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    # -1 for file name
    # -2 for its folder and proceed so on
    label = imagePath.split(os.path.sep)[-2]  # 	folder name withmask/withoutmask
    print(label)

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)  # corrects size to [noofImages, w, h, 1]

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
