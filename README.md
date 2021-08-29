# Contents

 - [intro](#introduction)
 - [use](#usage)

# Introduction 

It takes YOLO format dataset, performs the mosaic augmentation and saves data in yolo format (whose format has been tested later with voxelFiftyOne)


# Usage

Generate the mosaic augmented images with each image containing 4 images at different scales.

> `python main.py` 

Then replace the paths in the images.txt file with absolute paths.

> `python generate_train._txt.py`


Correct `obj.names` file manually.