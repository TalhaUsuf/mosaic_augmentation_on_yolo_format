# ==========================================================================
#                             voxel made COCO doesnot contain 
#  "is_crowd" key so this script is used to add this key to the json file                                  
# ==========================================================================

from pathlib import Path
from rich.console import Console
import json
from rich.pretty import pprint
from tqdm import tqdm

class paths(object):
    test = "/home/talha/Downloads/w8forms/test_coco/labels.json"
    train = "/home/talha/Downloads/w8forms/train_coco/labels.json"


# paths_ = list([paths.test , paths.train])
# pprint(paths_)


for k in tqdm([paths.train , paths.test] ,total=2, desc="FILE No:", colour="magenta") :
        
    annots = json.load(open(Path(k), "r"))

    for annot in tqdm(annots["annotations"], total=len(annots["annotations"]), desc="Annot. No:", colour="green"):
        annot["iscrowd"] = 0
        annot["segmentation"] = None
        #annot["keypoints"] = []


        
    json.dump(annots, open(Path(k), "w"), indent=4)



