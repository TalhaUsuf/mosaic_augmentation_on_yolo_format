from pathlib import Path
from absl import app, flags
from absl.flags import FLAGS
from rich.console import Console
from shutil import copy2, rmtree
from rich.progress import Progress, track
import os
from rich.table import Table


flags.DEFINE_list('src', default=["original_w9_augmented", "yolo_augmented"], help='list of relative paths of yolo dataset dirs. which need to be combined into single yolo dataset')
flags.DEFINE_string('dst', default="combined_yolo_ds", help='relative path of destination directory inside which all datasets will be combined in yolo format')




def confirm_obj_files(f1 : str, f2 : str):
    '''
    confirms both obj.names files are same

    Parameters
    ----------
    f1 : str
        path of the ist obj.names file
    f2 : str
        path of the second obj.names file
    '''    
    with open(f1, 'r') as f:
        f1_content = [k.strip() for k in f.readlines()]
        
    with open(f2, 'r') as f:
        f2_content = [k.strip() for k in f.readlines()]

    assert f1_content == f2_content, "obj.names files from both datasets are not same"

    tab = Table(title="obj.names contents", title_style="bold red on yellow", show_lines=True, style="green on black")
    tab.add_column(f"{f1}", justify="center", style="bold yellow")
    tab.add_column(f"{f2}", justify="center", style="bold cyan")
    tab.add_row(f"{f1_content}", f"{f2_content}")

    Console().print(tab)


def main(argv):

    
    dst = Path(FLAGS.dst)
    src = FLAGS.src # will be a list
    sources = [Path(k) for k in src if Path(k).exists()] # only select those input yolo dirs. which actually exist on disk
    assert len(sources) == 2, "there should be 2 yolodataset dirs. which need to be combined"
    Console().rule(title=f'[bold cyan]Combining [bold green]{sources[0]}\n{sources[1]}', characters='-', style='bold yellow')

    # check if the combined_yolo destination exists
    # if exists --> delete , then recreate dir.
    if dst.exists():
        rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    # confirm both the obj.names files are same
    confirm_obj_files(str(sources[0] / "obj.names"), str(sources[1] / "obj.names"))
    Console().rule(title=f'[color(128)]confirmed all obj.names are  [bold red]same', characters='-', style='bold magenta')

    # copy any one obj.names file to the combined_yolo dir
    copy2(str(sources[0] / "obj.names"), str(dst))

    #  make an images.txt file inside the combined_yolo dir
    (dst / "images.txt").touch()

    # make a data dir inside the combined_yolo dir
    (dst / "data").mkdir(parents=True, exist_ok=True)

    # copy all from sources to dst
    
    with Progress() as progress:
        t1 = progress.add_task(f"{sources[0].as_posix()}", total=list((sources[0] / "data").iterdir()).__len__())
        t2 = progress.add_task(f"{sources[1].as_posix()}", total=list((sources[1] / "data").iterdir()).__len__())
        for f in (sources[0] / "data").iterdir():
            if f.is_file():
                copy2(str(f), str(dst / "data"))
                progress.update(t1, advance=1)
        for f in (sources[1] / "data").iterdir():
            if f.is_file():
                copy2(str(f), str(dst / "data"))
                progress.update(t2, advance=1)

    
    with open(str(dst / "images.txt"), "w") as ff:
        for f in track((dst / "data").iterdir(), total = list((dst / "data").iterdir()).__len__(), description="Writing imgs/annots."):
            if f.is_file() and not f.name.endswith(".txt"): # .txt files are not written in images.txt file
                ff.write(f"data/{f.name}\n")            


if __name__ == '__main__':
    app.run(main)