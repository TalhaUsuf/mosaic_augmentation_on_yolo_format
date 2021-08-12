from pathlib import Path
import sys


dataset_dir = Path(".") / "w9"


print(f"dataset_dir is ===> {dataset_dir}")
file = dataset_dir / "train.txt"
# sys.stdout = f
img_dir = Path(dataset_dir ,"data")

images = list(img_dir.glob("*.png"))

images = ["data/"+str(k.name)+"\n" for k in images]
with open(file.as_posix(), "w") as f:
    f.writelines(images)
