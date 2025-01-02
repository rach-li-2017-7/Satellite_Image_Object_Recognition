# Product Overview

We use deep learning to mark the progress of construction work based on satellite images, to provide the land developers a timely and accurate project update, which they’d otherwise spend $xxk/month to check.

________________________________________

# Our Approach 

We used Semantic Segmentation to categorize the pixels of the satellite image into four categories: buildings, roads, vegetation, and everything else. 

## In-Scope

Given the time limit of Sundai's time, we used models to detect objects in satelitte images, and segmented them in pixel level. 

## Out-of-Scope 

If in a corporation setting and given more time, we could try either use Image with ground truth + Train the ML model, or use existing ML models + Test and Fine-tune the model. We would also work on the function to show the construction progress based on our detection results.

________________________________________

# Our Accomplishment

Explored prior work and existing tools and chose the approach;

Created GitHub and Google Shared Drive to collaborate work;

Prepared satellite image in GeoTIFF format, and a CAD design of the same jobsite in GeoJSON format.

1. Data Preparation: Transform the GeoTIFF satellite images to JPG format, ready for Deep Learning model.
2. Leveraged Inception Resnetv2 Unet to do sementic segmentation based on processed satelitte image data, output segementation results as masks into PNG format.
3. Combined original satelitte image, CAD image, and predicted masks, show final visualization results.

# Limitations

- Since we used pretrained models, the arrcuracy is not ideal.
- Limited images are available now.

# Next Steps

- Convert the output of Deep Learning model to KML or SHP format.
- Fine-tune models based on images that better suits our use case.
- Determine metrics to calculate the percentage of works that have been done by workers.
- Create smaller granularity in detection.

# References

1. **[dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation)**
