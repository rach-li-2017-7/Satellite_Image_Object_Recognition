# import packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.utils import get_label_mappings, onehot_to_rgb, filter_mask_colors, convert_to_transparent_png, mask_images
from tensorflow.keras.models import load_model as keras_load_model
from utils.utils import build_inception_resnetv2_unet

def load_model():
    # Load the InceptionResNetV2-UNet model and its weights
    model = build_inception_resnetv2_unet(input_shape=(512, 512, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights("model/InceptionResNetV2-UNet.h5")
    return model

def process_and_save_masks(original_image_path, cad_image_path, model):
    class_dict_df, code2id, id2code, name2id, id2name = get_label_mappings()

    # Load and preprocess the original image
    image = cv2.imread(original_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (512, 512))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0

    # Make prediction with the pre-loaded model
    predicted_mask = model.predict(image_array)

    # Convert the predicted mask to RGB
    predicted_mask_rgb = onehot_to_rgb(predicted_mask[0], id2code)

    # Resize the predicted mask back to the original image dimensions
    original_height, original_width = image.shape[:2]
    predicted_mask_resized = cv2.resize(predicted_mask_rgb, 
                                         (original_width, original_height),
                                         interpolation=cv2.INTER_NEAREST)

    # Filter the mask to keep only red (buildings) and blue (roads)
    colors_to_keep = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
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
