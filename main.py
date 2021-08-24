'''make 'data' folder inside w9 folder and place images and respective annotations inside 'data' folder'''
import random
from rich.console import Console
import cv2
import os
import shutil
import glob
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from rich.console import Console

OUTPUT_SIZE = (1000, 1000)  # Height, Width
# SCALE_RANGE = (0.3, 0.7)
SCALE_RANGE = (0.3, 1.0)
# FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.
FILTER_TINY_SCALE = -1  # if height or width lower than this scale, drop it.

ANNO_DIR = './dataset/WiderPerson/Annotations/'
IMG_DIR = './dataset/WiderPerson/Images/'

category_name = ['background', 'person']


def main():
    img_paths, annos = get_dataset(ANNO_DIR, IMG_DIR)
    if os.path.exists("yolo_data/data"):
        shutil.rmtree("yolo_data/data")
    Path("yolo_data/data").mkdir(exist_ok=True, parents=True)
    for idx in tqdm(range(1000), desc="AUGMENTATION-NO:", total=1000):

        # sample randomly indices out of a list of 22 annotations
        idxs = random.sample(range(len(annos)), 4)

        new_image, new_annos = update_image_and_anno(img_paths, annos,
                                                     idxs,
                                                     OUTPUT_SIZE, SCALE_RANGE,
                                                     filter_scale=FILTER_TINY_SCALE)
        copy_img = copy.deepcopy(new_image)
        # name = np.random.randint(0, 1000, (1,)).squeeze()
        name = idx
        # cv2.imwrite(f'./data/{idx}_without_boxes.jpg', copy_img)
        cv2.imwrite(f'yolo_data/data/{name}.jpg', copy_img)
        Console().print(f"after augmenting, {len(new_annos)} annotations made .... ")
        # f = open(f'./data/{idx}_without_boxes.txt', 'w')
        f = open(f'yolo_data/data/{name}.txt', 'w')
        colors = np.random.randint(0, 255, (len(new_annos), 3))
        colors = [tuple([int(k) for k in j]) for j in colors] # list of tuples of colors
        lines = []
        for k, anno in enumerate(new_annos):
            # all augmented annotations generated for this image
            # c = random.sample(range(255), 3)
            # c = tuple([int(k) for k in c])
            start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
            end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
            # cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)
            new_image = cv2.rectangle(new_image, start_point, end_point, colors[k], 2, cv2.LINE_AA)
            # convert wider-person back to yolo format
            cpy_annot = copy.deepcopy(anno)
            cpy_annot[0] = anno[0]
            cpy_annot[1] = ((start_point[0] + end_point[0]) / 2) / OUTPUT_SIZE[1]
            cpy_annot[2] = ((start_point[1] + end_point[1]) / 2) / OUTPUT_SIZE[0]
            cpy_annot[3] = (end_point[0] - start_point[0]) / OUTPUT_SIZE[1]
            cpy_annot[4] = (end_point[1] - start_point[1]) / OUTPUT_SIZE[0]


            f.write(f"{cpy_annot[0]} {cpy_annot[1]} {cpy_annot[2]} {cpy_annot[3]} {cpy_annot[4]}\n")
            # lines.append(' '.join([str(i) for i in cpy_annot]))
        # f.writelines(lines)
        # cv2.imwrite(f'data/{idx}_with_boxes.jpg', new_image)
        f.close()
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                     disable showing images
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        # new_image = Image.fromarray(new_image.astype(np.uint8))
        # new_image.show()


def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    # divid_point is the (x,y) point of intesection
    #                                                             [  0  ,    1  ]
    divid_point_x = int(scale_x * output_size[1]) # output_size : [rows, columns]
    divid_point_y = int(scale_y * output_size[0])


    new_anno = []
    for i, idx in enumerate(idxs): # idxs is a list of 4 random indices
        path = all_img_list[idx] # select a single (image,annotation) at index `idx`
        img_annos = all_annos[idx]

        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * scale_x
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = bbox[2] * scale_y
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = bbox[3] * scale_x
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
    #                               0      1      2     3     4
    if 0 < filter_scale: # anno : [label, xmin, ymin, xmax, ymax]
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    return output_img, new_anno


def get_dataset(anno_dir, img_dir):

    class_id = category_name.index('person')

    img_paths = []
    annos = []
    base_path = Path(".")  # current dir
    # images = base_path / "dataset" / "WiderPerson" / "Images"
    # annotations = base_path / "dataset" / "WiderPerson" / "Annotations"
    num_classes = len([k for k in open(base_path / "w9" / "obj.names").readlines() if not k.isspace()])
    Console().print(f"total classes are : ====> {num_classes}")
    images = base_path / "w9" / "data"
    annotations = base_path / "w9" / "data"
    img_dir = base_path / "w9" / "data"
    # Console().print(f"image path==>{images}")
    # Console().print(f"annotations path==>{annotations}")
    # Console().rule(title="paths", align="center", style="cyan")
    annot_paths = list(annotations.glob("*.txt"))
    # for anno_file in glob.glob(os.path.join(anno_dir, '*.txt')):
    for anno_file in annot_paths:
        anno_file = anno_file.as_posix()
        anno_id = anno_file.split('/')[-1].split('.')[0]

        with open("./" + anno_file, 'r') as f:


            # num_of_objs = int(f.readline())
            with open("./" + anno_file, 'r') as ff:
                num_of_objs = len([k.strip() for k in ff.readlines() if not k.isspace()]) # f.readlines() --> list
            Console().print(f"{num_of_objs} annotations are found ... ")
            # img_path = os.path.join(img_dir, f'{anno_id}.jpg')
            # Console().print(f"current dir. ===> {Path().cwd()}")
            img_path = Path("./"+img_dir.as_posix(),f'{anno_id}.png').as_posix()
            # Console().print(f"image path is : {img_path}")
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
            del img

            boxes = []
            for _ in range(num_of_objs):
                obj = f.readline().rstrip().split()
                obj = [float(elm) for elm in obj]
                label_idx = int(obj[0])
                if num_classes < label_idx: # skip annotation line if class idx is > no. of classes
                    continue
                # convert annotations from yolo to wider-person format
                #         [ 0      1    2     3  4]
                # convert [label, xcen, ycen, w, h] --> all normalized
                # replace the coordinates                      [     0       1   2    3   4]
                                             # obj_format ---> [class_label, x1, y1, x2, y2]

                # convert yolo to wider-person annotation format
                xmin = (obj[1] - (obj[3] / 2)) * img_width
                ymin = (obj[2] - (obj[4] / 2)) * img_height
                xmax = (obj[1] + (obj[3] / 2)) * img_width
                ymax = (obj[2] + (obj[4] / 2)) * img_height

                obj[1] = xmin
                obj[2] = ymin
                obj[3] = xmax
                obj[4] = ymax

                xmin = max(obj[1], 0) / img_width
                ymin = max(obj[2], 0) / img_height
                xmax = min(obj[3], img_width) / img_width
                ymax = min(obj[4], img_height) / img_height

                boxes.append([label_idx, xmin, ymin, xmax, ymax])

            if not boxes:
                continue
        img_paths.append(img_path) # img_path is a list of images
        annos.append(boxes)
    with Console().status("lists made ....", spinner="bouncingBall"):
        Console().print(f"{img_paths}")
        # Console().print(f"{annos}")
    return img_paths, annos


if __name__ == '__main__':
    main()
