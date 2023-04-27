import tarfile
import os
import gdown
import shutil

dir_path = "./data/CUB_200_2011"
files = {
    'dataset': {
        'url': "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        "path": "CUB-200-2011.tgz"
    },
    'segmentations': {
        'url': "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1",
        "path": "segmentations.tgz"
    }
}

os.makedirs(dir_path, exist_ok=True)
for name in files:
    file_path = os.path.join(dir_path, files[name]['path'])
    gdown.download(files[name]['url'], file_path, quiet=False)
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(path='./data')
    tar.close()
shutil.move(os.path.join('./data', 'segmentations'), os.path.join(dir_path, 'segmentations'))