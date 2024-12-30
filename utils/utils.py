import keract

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


def get_label_mappings(csv_path='input/class_dict.csv'):
    """
    Create mappings between label codes, IDs and names from class dictionary CSV.
    
    Args:
        csv_path: Path to CSV file containing class definitions
        
    Returns:
        code2id: Dict mapping RGB codes to class IDs
        id2code: Dict mapping class IDs to RGB codes  
        name2id: Dict mapping class names to IDs
        id2name: Dict mapping class IDs to names
    """
    class_dict_df = pd.read_csv(csv_path, index_col=False, skipinitialspace=True)
    
    label_names = list(class_dict_df.name)
    label_codes = []
    r = np.asarray(class_dict_df.r)
    g = np.asarray(class_dict_df.g) 
    b = np.asarray(class_dict_df.b)
    
    for i in range(len(class_dict_df)):
        label_codes.append(tuple([r[i], g[i], b[i]]))

    code2id = {v:k for k,v in enumerate(label_codes)}
    id2code = {k:v for k,v in enumerate(label_codes)}
    
    name2id = {v:k for k,v in enumerate(label_names)}
    id2name = {k:v for k,v in enumerate(label_names)}
    
    return code2id, id2code, name2id, id2name


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_inception_resnetv2_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained InceptionResNetV2 Model """
    encoder = InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = encoder.get_layer("input_1").output           ## (512 x 512)

    s2 = encoder.get_layer("activation").output        ## (255 x 255)
    s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)         ## (256 x 256)

    s3 = encoder.get_layer("activation_3").output      ## (126 x 126)
    s3 = ZeroPadding2D((1, 1))(s3)                     ## (128 x 128)

    s4 = encoder.get_layer("activation_74").output      ## (61 x 61)
    s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)           ## (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer("activation_161").output     ## (30 x 30)
    b1 = ZeroPadding2D((1, 1))(b1)                      ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)
    
    """ Output """
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(6, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs, name="InceptionResNetV2-UNet")
    return model


def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def filter_mask_colors(mask, colors_to_keep):
    """
    Filter a mask to keep only specified colors, setting all others to black.
    
    Args:
        mask: numpy array of shape (height, width, 3) containing RGB values
        colors_to_keep: list of RGB tuples to preserve, e.g. [(255, 0, 0), (0, 0, 255)]
    
    Returns:
        filtered_mask: numpy array with same shape as input, with only specified colors preserved
    """
    filtered_mask = mask.copy()
    
    # Create combined boolean mask for all colors to keep
    combined_mask = np.zeros(mask.shape[:2], dtype=bool)
    for color in colors_to_keep:
        color_mask = np.all(filtered_mask == color, axis=-1)
        combined_mask = combined_mask | color_mask
    
    # Set all other pixels to black
    filtered_mask[~combined_mask] = [0, 0, 0]
    
    return filtered_mask


def convert_to_transparent_png(mask, output_path):
    """
    Convert an RGB mask to RGBA PNG with black (0,0,0) as transparent.
    
    Args:
        mask: numpy array of shape (height, width, 3) containing RGB values
        output_path: string, path where to save the PNG file
    """
    # Convert to RGBA with black as transparent
    rgba_mask = np.zeros((*mask.shape[:2], 4))
    rgba_mask[..., :3] = mask / 255.0  # Normalize RGB values to 0-1 range
    
    # Set alpha channel - make black pixels transparent
    is_black = np.all(mask == [0, 0, 0], axis=-1)
    rgba_mask[..., 3] = ~is_black  # Set alpha to 0 for black pixels, 1 for others
    
    # Save as PNG with transparency
    plt.imsave(output_path, rgba_mask)


def mask_images(image_path, mask_path, target_size):
    """
    Load and resize original image and filtered mask to target size.
    
    Args:
        image_path: Path to original image
        mask_path: Path to filtered mask image
        target_size: Tuple of (height, width) for resizing
    
    Returns:
        Tuple of (resized_original, resized_mask)
    """
    original_image = plt.imread(image_path)
    filtered_mask = plt.imread(mask_path)
    
    # Resize both images (width, height for cv2.resize)
    original_image = cv2.resize(original_image, target_size[::-1])
    filtered_mask = cv2.resize(filtered_mask, target_size[::-1])
    
    return original_image, filtered_mask

def display_image_with_mask(original_image, filtered_mask, figsize=(12,6)):
    """
    Display original image and masked overlay side by side.
    
    Args:
        original_image: Original image array
        filtered_mask: Mask image array
        figsize: Figure size tuple, default (12,6)
    """
    plt.figure(figsize=figsize)
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Overlay mask on original image
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(filtered_mask, alpha=0.5)
    plt.title('Image with Filtered Mask Overlay')
    plt.axis('off')
    
    plt.show()

