# import packages
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from IPython.display import SVG
import matplotlib.pyplot as plt
import os, re, sys, random, shutil, cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout
from tensorflow.keras.models import load_model

# import utils
from utils.utils import get_label_mappings, conv_block, decoder_block, build_inception_resnetv2_unet, rgb_to_onehot, onehot_to_rgb, dice_coef, filter_mask_colors, convert_to_transparent_png, mask_images, display_image_with_mask


def process_and_save_masks(original_image_path, cad_image_path):
    code2id, id2code, name2id, id2name = get_label_mappings()

    model = build_inception_resnetv2_unet(input_shape=(512, 512, 3))
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', 
                 metrics=[dice_coef, "accuracy"])

    model.load_weights("model/InceptionResNetV2-UNet.h5")

    # Load and preprocess the image
    image = cv2.imread(original_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (512, 512))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0

    # Make prediction
    predicted_mask = model.predict(image_array)

    # Convert the predicted mask to RGB
    predicted_mask_rgb = onehot_to_rgb(predicted_mask[0], id2code)

    # Resize the predicted mask back to the original image dimensions
    original_height, original_width = image.shape[:2]
    predicted_mask_resized = cv2.resize(predicted_mask_rgb, 
                                      (original_width, original_height),
                                      interpolation=cv2.INTER_NEAREST)

    # Filter the mask to keep only red (buildings) and blue (roads)
    colors_to_keep = [(255, 0, 0), (0, 0, 255)]
    filtered_mask = filter_mask_colors(predicted_mask_resized, colors_to_keep)

    # Save the filtered mask as transparent PNG
    convert_to_transparent_png(filtered_mask, 'output/filtered_mask.png')

    # Process original image
    target_size = (1280, 1920)
    original_image, filtered_mask = mask_images(
        original_image_path,
        'output/filtered_mask.png',
        target_size
    )

    # Save the overlaid result for original image
    plt.figure(figsize=(12, 6))
    plt.imshow(original_image)
    plt.imshow(filtered_mask, alpha=0.5)
    plt.axis('off')
    plt.savefig('output/result_origin.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Process CAD image
    original_image, filtered_mask = mask_images(
        cad_image_path,
        'output/filtered_mask.png',
        target_size
    )

    # Save the overlaid result for CAD image
    plt.figure(figsize=(12, 6))
    plt.imshow(original_image)
    plt.imshow(filtered_mask, alpha=0.5)
    plt.axis('off')
    plt.savefig('output/result_cad.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    process_and_save_masks("upload/SerenityCloseUp_v0_20241229.jpg", "upload/SerenityCloseUp_v0_20241229.jpg")