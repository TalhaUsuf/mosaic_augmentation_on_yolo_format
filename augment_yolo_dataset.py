# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      imports
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import albumentations as A
import cv2
from pathlib import Path
from rich.console import Console
import logging
from rich.progress import track
from rich.logging import RichHandler
from time import sleep
from ast import literal_eval
import numpy as np

# configure logs
FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
log = logging.getLogger("rich")

# read the dir. structure
Console().rule(characters="====")
base = Path("yolo_data")
# finding the obj.names file
obj_path = list(base.glob("*.names"))
assert len(obj_path) == 1, f"there should be only single obj.names path , found {obj_path}"
Console().log(f"obj.names at ===> {obj_path}")
log.info(f"obj_names file ---> {obj_path}")
objects = [k.strip() for k in open(obj_path[0].as_posix(), "r").readlines()]
idx2obj = dict(zip(range(len(objects)), objects))
obj2idx = {k: v for v, k in idx2obj.items()}
# Console().log(f"objects are : ====> {objects}")
# Console().log(f"idx2obj  : ====> {idx2obj}")
# Console().log(f"obj2idx  : ====> {obj2idx}")
log.info(f"objects are : ====> {objects}")
log.info(f"idx2obj are : ====> {idx2obj}")
log.info(f"obj2idx are : ====> {obj2idx}")
images_file = list(base.glob("*.txt"))
assert len(images_file) == 1, f"there should be only one images txt file"
log.info(f"txt files found ===> {images_file}")
images_list = [k.strip() for k in open(images_file[0].as_posix(), "r").readlines()]
log.info(f"IMAGES FOUND IN {images_file[0].as_posix()} FILE, \n {images_list}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      defining the augmentations
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
transform = A.Compose([
    A.RandomCrop(width=800, height=800),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['w9fields'], min_area=100, min_visibility=0.10))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                      loop over each image in
#             the images.txt file and read the corresponding annotations
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for img_path in track(images_list, description="PROCESSING IMG.", total=len(images_list), style="red on black"):
    log.info(f"processing image ---> {img_path}")
    parent = Path(img_path).parents[0]
    fname = Path(img_path).stem
    img = cv2.imread(parent.as_posix() + f"/{fname}.jpg")
    annot = [k.strip().split() for k in open(parent.as_posix() + f"/{fname}.txt", "r").readlines()]
    annot = [[float(m) for m in i] for i in annot]
    annot = [j for j in annot if not (j[3] < 1e-3 or j[4] < 1e-3)]  # remove annots with width and height = 0
    log.info(f"PREVIOUS ANNOTATIONS => {annot}")
    fields_idxs = [a.pop(0) for a in
                   annot]  # remove the bboxes from annot list (in-place op) and save class-idxs to fields_idxs
    log.info(f"CLASS-IDX REMOVED ANNOTATIONS => {annot}")
    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # #                      plot to see if boxes are correctly read
    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # H, W = img.shape[:2]
    # colors = np.random.randint(0, 255, (len(idx2obj), 3), dtype=int).astype(
    #     int)  # colors list with as many rows as classes
    # for k in annot:
    #     #         ['3', '0.3285', '0.1055', '0.191', '0.145'],
    #     cls, xcen, ycen, w, h = int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])
    #     xmin = int((xcen - w / 2) * W)
    #     ymin = int((ycen - h / 2) * H)
    #     xmax = int((xcen + w / 2) * W)
    #     ymax = int((ycen + h / 2) * H)
    #     c = tuple([int(k) for k in colors[cls]])
    #     log.info(c)
    #     img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), c, 1, cv2.LINE_AA)
    #     img = cv2.putText(img, f"{idx2obj[cls]}", (xmin, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, c, 1, cv2.LINE_AA)
    #     cv2.imshow("image", img)
    #     cv2.waitKey(10)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    w9fields = [idx2obj[k] for k in fields_idxs]
    log.info(f"w9fields ---> \n {w9fields}")
    transformed = transform(image=img, bboxes=annot, w9fields=w9fields)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    w9fields = transformed['w9fields']
    Console().print("%-35s %-s" % ("TRF. IMG SHAPE", transformed_image.shape))
    Console().print("%-35s %-s" % ("TRF. BOXES", transformed_bboxes))
    Console().print("%-35s %-s" % ("W9 FIELDS", w9fields))
