# Contents

 - [intro](#introduction)
 - [use](#usage)

# Introduction 

It takes YOLO format dataset, performs the mosaic augmentation and saves data in yolo format (whose format has been tested later with voxelFiftyOne)


# Usage

 - Generate the mosaic augmented images with each image containing 4 images at different scales. It takes [w9 format](w9) original dataset which is formed after hand annotations from the **cvat annotation** app. and saves the results to  [yolo_data](yolo_data) folder in yolo format.

> `python main.py` 

 - Run below script to update the paths in [./yolo_data/images.txt](yolo_data/images.txt) file.

> `python generate_train._txt.py`

 - Run following command to take [mosaic augmented dataset](yolo_data) and apply different image / bbox level augmentations from albumentations library. Resulting dataset will be saved to the [yolo_augmented](yolo_augmented) directory. 

> `python augment_yolo_dataset.py --img_type jpg`

 - Run following command to apply augmentations to the original hand annotated dataset:
> `python augment_yolo_dataset.py --naug 30 --path w9 --out original_w9_augmented --img_type png --out_img_type jpg`


# Validity of YOLO Dataset Format

Use the **voxel-fiftyone api** to find the validity of dataset. It can also be used to convert between different detection dataset formats and export on disk.  
> `fiftyone datasets create -n <custom name of dataset>  -d  <path to yolo format dataset folder>  -t fiftyone.types.YOLODataset --shuffle`

Run the fiftyone web app to visualize if annotaiton labels are correct:

> `fiftyone app launch`

# Note

The script [augment_yolo_dataset](augment_yolo_dataset.py) supports following flags:
```html
  --naug: no. of augmentations to apply for each image.
    (default: '10')
    (an integer)
  --out: path of the folder in which to save the augmented dataset
    (default: 'yolo_augmented')
  --path: yolo format dataset folder path. It must contain obj.names, images.txt files and data folder
    (default: 'yolo_data')

```