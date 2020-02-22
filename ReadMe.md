# Generate Synthetic Data for Amodal Segmentation and Completion
To get ground truth for occlusion objects, we can synthesis data with part of it gets occluded by other objects. 

## class of possible occlusion

## output spec
- four folders: 
    1. full 
    2. full mask
    3. occlusion
    4. occlusion mask
   
    the first two are raw data. 
    the second two are target synthetic data. 
- image size: 128 * 128
- For a single synthetic data, we want the output to be centered and scale to fill the whole image.

### transform 
translation, rotation, scale

### pipeline
1. get mask1, img1; mask2, img2
2. random transform on mask2
3. combine mask1 and mask2. make sure two masks have overlap
4. mask2 area becomes black
5. move union of two mask areas to the center and rescale

### preprocess
know what is 

