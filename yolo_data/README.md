# Table Of Contents

 - [Introduction](#introduction)
 - [Dir. Structure](#directory-structure)
 - [How is this dataset produced](#how-this-dataset-is-produced)
 - [Check Dataset validity](#validity-of-yolo-dataset-format)


# Introduction

This is the **mosaic augmented dataset** on which mosaic augmentation has been applied.
This dataset must be in the **YOLO** format. The details of this dataset format can be found in the link below:


> [Yolo Dataset Format](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#yolov4dataset)

# How this dataset is produced

This dataset has been applied by running [this](../main.py) script which takes [this](../w9) yolo format dataset (original dataset), applies the **mosaic** augmentation and dumps results into this directory.
The script tiles 4 images in single image and those 4 images are randomly chosen.

# Directory Structure

The directory structure is given below. 

```html
yolo_data
├── data
│   ├── 0.jpg
│   ├── 0.txt
│   ├── 10.jpg
│   ├── 10.txt
│   ├── 11.jpg
│   ├── 11.txt
│   ├── 12.jpg
│   ├── 12.txt
│   ├── 13.jpg
│   ├── 13.txt
│   ├── 14.jpg
|   .
|   .
|   .
|   .
|   .
|   .
│   ├── 97.txt
│   ├── 98.jpg
│   ├── 98.txt
│   ├── 99.jpg
│   ├── 99.txt
│   ├── 9.jpg
│   └── 9.txt
├── images.txt
├── obj.names
└── README.md
```

Conditions for a valid dataset are given below:

 - `obj.names` file must be there. **Order of classes is important**
 - `images.txt` file must be present. It may contain **relative** or **absolute paths**
 - `data` folder must be there containing the images and txt files with same file-names

# Validity of YOLO Dataset Format

Use the **voxel-fiftyone api** to find the validity of dataset. It can also be used to convert between different detection dataset formats and export on disk.  
> `fiftyone datasets create -n yolo_mosaic_augmented_w9  -d  yolo_data  -t fiftyone.types.YOLODataset --shuffle`

Run the fiftyone web app to visualize if annotaiton labels are correct:

> `fiftyone app launch`