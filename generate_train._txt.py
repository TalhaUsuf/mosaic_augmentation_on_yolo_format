from pathlib import Path
import sys


dataset_dir = Path(".") / "yolo_data"


print(f"dataset_dir is ===> {dataset_dir}")
file = dataset_dir / "images.txt"
# sys.stdout = f
img_dir = Path(dataset_dir , "data")

images = list(img_dir.glob("*.jpg"))

# images = ["data/"+str(k.name)+"\n" for k in images]
images = [(dataset_dir.absolute() / "data").as_posix()+"/"+str(k.name)+"\n" for k in images]
with open(file.as_posix(), "w") as f:
    f.writelines(images)
