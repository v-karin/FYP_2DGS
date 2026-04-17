import os
from os import path as osp
import shutil

from kagglehub import dataset_download
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

data_path = osp.join(osp.dirname(__file__), "..", "data")
dl_path = osp.join(data_path, "raw")
preproc_path = osp.join(data_path, "preproc")




def move_imgs(src: str, dst: str, rename_func=lambda x: x):
    os.makedirs(dst, exist_ok=True)

    for file in os.listdir(src):
        os.rename(osp.join(src, file), osp.join(dst, rename_func(file)))


def move_train_test(name: str):
    src_path = osp.join(dl_path, name)
    dst_path = osp.join(preproc_path, name)

    move_imgs(osp.join(src_path, "test"), dst_path, lambda s: "te_" + s)
    move_imgs(osp.join(src_path, "train"), dst_path, lambda s: "tr_" + s)


def move_imgs_basic(name: str):
    src_path = osp.join(dl_path, name)
    dst_path = osp.join(preproc_path, name)

    move_imgs(src_path, dst_path)




dataset_profiles = {
    #"artworks": "ikarus777/best-artworks-of-all-time",
    "butterfly": {
        "link": "phucthaiv02/butterfly-image-classification",
        "preproc": move_train_test,
    },
    "kodak": {
        "link": "sherylmehta/kodak-dataset",
        "preproc": move_imgs_basic,
    }
}




def download(name: str):
    out = dataset_download(dataset_profiles[name]["link"], output_dir=osp.join(dl_path, name))
    print(f"Downloaded to:\n{out}")


def preproc(name: str):
    dataset_profiles[name]["preproc"](name)




class ImageLoader(Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = transforms.PILToTensor()
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path = osp.join(self.root, self.images[index])
        return self.transform(Image.open(path).convert("RGB"))




def load_data(name: str):
    "Can use any torchvision ImageFolder kwargs"
    path = osp.join(preproc_path, name)
    if not osp.exists(path):
        download(name)
        preproc(name)

    return ImageLoader(path)