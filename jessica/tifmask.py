import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Step 1: Load the GeoTIFF file
file_path = "SerenitySatelliteImage_v0_20241222.tif"

# Open the GeoTIFF file using rasterio
with rasterio.open(file_path) as src:
    # Read the first band as an example (modify based on your file structure)
    segmentation_mask = src.read(1)  # Read the first band (class indices)

# Step 2: Define the class-to-color mapping
class_colors = {
    0: [0, 0, 0],  # Black for background
    1: [255, 0, 0],  # Red for class 1
    2: [0, 255, 0],  # Green for class 2
    3: [0, 0, 255],  # Blue for class 3
    4: [255, 255, 0],  # Yellow for class 4
}

# Step 3: Convert the mask to an RGB image
rgb_mask = np.zeros((*segmentation_mask.shape, 3), dtype=np.uint8)
for class_id, color in class_colors.items():
    rgb_mask[segmentation_mask == class_id] = color

# Step 4: Visualize the RGB mask
plt.figure(figsize=(10, 10))
plt.imshow(rgb_mask)
plt.title("RGB Segmentation Mask")
plt.axis("off")
plt.show()

# Step 5: Optional - Visualize using a colormap for class indices
cmap = ListedColormap(["black", "red", "green", "blue", "yellow"])  # Match class_colors
plt.figure(figsize=(10, 10))
plt.imshow(segmentation_mask, cmap=cmap)
plt.colorbar(label="Class Index")
plt.title("Colormap Segmentation Mask")
plt.axis("off")
plt.show()
