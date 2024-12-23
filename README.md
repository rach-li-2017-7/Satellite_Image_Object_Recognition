# Satellite_Image_Object_Recognition
## Product Overview

We use deep AI to mark the progress of construction work on satellite images, to provide the land developers a timely and accurate project update, which they’d otherwise spend $xxk/month to check.
________________________________________
## Our Approach 

We plan to use Semantic Segmentation to categorize the pixels of the satellite image into four categories: road, house, grass, everything else. We do not use Object detection, because our goal is not to recognize the object but to measure the length and area of them and to compare with the design.

### In-Scope

Given the time limit of Sundai, we plan to find existing APIs to realize the Semantic Segmentation.

### Out-of-Scope 

If in a corporation setting and given more time, we could try either
	use Image with ground truth + Train the ML model;
	or use existing ML models + Test and Fine-tune the model.
________________________________________
## Our Accomplish Today

Formed a team with experts;

Explored prior work and existing tools and chose the approach;

Created GitHub and Google Shared Drive to collaborate work;

Prepared satellite image in GeoTIFF format, and a CAD design of the same jobsite in GeoJSON format.
________________________________________
## What We Plan to Do 

•	Transform the GeoTIFF satellite images to a pixel format, ready for Deep Learning model

•	Find and construct a suitable Deep Learning model to do Semantic Segmentation

•	Convert the output of Deep Learning model to KML or SHP format

•	Create a web visualization tool to present the result
