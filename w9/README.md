# Table Of Contents

 - [Introduction](#introduction)
 - [Dir. Structure](#directory-structure)
 - [Check Dataset validity](#validity-of-yolo-dataset-format)

# Introduction

This is the **original dataset** on which augmentation needs to be performed.
This dataset must be in the **YOLO** format. The details of this dataset format can be found in the link below:


> [Yolo Dataset Format](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#yolov4dataset)

# Directory Structure

The directory structure is given below. 
```w9
├── data
│   ├── W9_form_page1.png
│   ├── W9_form_page1.txt
│   ├── W9_form_page2.png
│   ├── W9_form_page2.txt
│   ├── W9_form_page3.png
│   ├── W9_form_page3.txt
│   ├── W9_form_page4.png
│   ├── W9_form_page4.txt
│   ├── W9_form_page5.png
│   ├── W9_form_page5.txt
│   ├── W9_form_page6.png
│   └── W9_form_page6.txt
├── obj.data
├── obj.names
├── README.md
└── train.txt
```

Conditions for a valid dataset are given below:

 - `obj.names` file must be there. **Order of classes is important**
 - `images.txt` file must be present. It may contain **relative** or **absolute paths**
 - `data` folder must be there containing the images and txt files with same file-names

# Validity of YOLO Dataset Format

Use the **voxel-fiftyone api** to find the validity of dataset. It can also be used to convert between different detection dataset formats and export on disk.  
> `fiftyone datasets create -n yolo_augmented_w9  -d  yolo_augmented  -t fiftyone.types.YOLODataset --shuffle`

Run the fiftyone web app to visualize if annotaiton labels are correct:

> `fiftyone app launch`