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
import os
import itertools
import shutil
from tqdm import trange, tqdm
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string("path", default="yolo_data",
                    help="yolo format dataset folder path. It must contain obj.names, images.txt files and data folder ")
flags.DEFINE_string("out", default="yolo_augmented", help="path of the folder in which to save the augmented dataset")
flags.DEFINE_integer("naug", default=10, help="no. of augmentations to apply for each image.")
flags.DEFINE_string("img_type", default="jpg", help="Extension of the images in yolodataset , it can be png, jpeg jpg, See the 'data' dir. to see in which format are images present.")
flags.DEFINE_string("out_img_type", default="jpg", help="Images after augmentation will also be saved in format specified by this flag. It can be png, jpeg jpg")

def main(argv):
    # configure logs
    FORMAT = "%(message)s"
    logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
    log = logging.getLogger("rich")

    # read the dir. structure
    Console().rule(characters="====")
    base = Path(FLAGS.path)
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
        A.RandomCrop(width=800, height=800, p=0.5),
        A.RandomRotate90(p=0.8),
        A.Rotate(limit=5, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.ISONoise(p=0.5),
        A.CLAHE(clip_limit=5.0, tile_grid_size=(8,8), p=0.5),
        A.ColorJitter(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(height=1000, width=1000, always_apply=True),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['w9fields'], min_area=100, min_visibility=0.05))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                      loop over each image in
    #             the images.txt file and read the corresponding annotations
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if os.path.exists(f"./{FLAGS.out}"):
        shutil.rmtree(f"./{FLAGS.out}")  # delete the existing dir.
        Path(f"{FLAGS.out}/data").mkdir(exist_ok=True, parents=True)  # make new dir.

    Path(f"{FLAGS.out}/data").mkdir(exist_ok=True, parents=True)  # make new dir.
    for img_path in track(images_list, description="PROCESSING IMG.", total=len(images_list), style="red on black"):
        log.info(f"processing image ---> {img_path}")
        parent = Path(img_path).parents[0]
        fname = Path(img_path).stem
        img = cv2.imread(FLAGS.path +"/"+ parent.as_posix() + f"/{fname}.{FLAGS.img_type}")
        annot = [k.strip().split() for k in open(FLAGS.path +"/"+ parent.as_posix() + f"/{fname}.txt", "r").readlines()]
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
        for aug_no in trange(FLAGS.naug, desc="Augmentation No.", leave=False):
            # for aug_no in range(2):
            transformed = transform(image=img, bboxes=annot, w9fields=w9fields)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            w9fields_trf = transformed['w9fields']
            assert len(w9fields_trf) == len(
                transformed_bboxes), f"length of augmented bboxes and augmented labels should match each other"
            Console().print("%-35s %-s" % ("TRF. IMG SHAPE", transformed_image.shape))
            Console().print("%-35s %-s" % ("TRF. BOXES", transformed_bboxes))
            Console().print("%-35s %-s" % ("W9 FIELDS", w9fields_trf))

            #         writing to the file
            #     single image , multi - annotations , multi - labels
            trf_fields2idxs = [obj2idx[k] for k in w9fields_trf]
            shutil.copyfile(obj_path[0].as_posix(),
                            f"{FLAGS.out}/" + f"{obj_path[0].name}")  # copy the objk.names file to the augmented images directory
            # write the image to dir.
            cv2.imwrite(f"./{FLAGS.out}/data/{fname}_{aug_no}.{FLAGS.out_img_type}", transformed_image)
            with open(f"./{FLAGS.out}/data/{fname}_{aug_no}.txt", "w") as faug:
                for label, box in tqdm(zip(trf_fields2idxs, transformed_bboxes), total=len(trf_fields2idxs),
                                       colour="green", desc="WRITING TO FILE"):
                    faug.write(str(label) + " " + " ".join([str(k) for k in list(box)]) + "\n")

            # annotation file writing completed

    with open(f"{FLAGS.out}/images.txt", "w") as fimgs:
        imgs = list(Path(f"{FLAGS.out}/data").glob(f"*.{FLAGS.out_img_type}"))
        imgs = [k.parents[0].stem + f"/{k.name}" "\n" for k in imgs]
        fimgs.writelines(imgs)


if __name__ == "__main__":
    app.run(main)
