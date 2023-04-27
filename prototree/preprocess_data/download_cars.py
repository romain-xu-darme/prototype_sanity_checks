import tarfile
import os
import gdown

dir_path = "./data/cars"
files = {
    'trainset': {
        'url': "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
        'path': 'cars_train.tgz'
    },
    'testset': {
        'url': "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
        'path': 'cars_test.tgz'
    },
    'devkit': {
        'url': "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
        'path': 'cars_devkit.tgz'
    },
    'test_anno': {
        'url': "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
        'path': 'cars_test_annos_withlabels.mat'
    },
    'bb_anno': {
        'url': 'http://ai.stanford.edu/~jkrause/car196/cars_annos.mat',
        'path': 'cars_annos.mat',
    }
}

os.makedirs(dir_path, exist_ok=True)
for name in files:
    file_path = os.path.join(dir_path, files[name]['path'])
    gdown.download(files[name]['url'], file_path, quiet=False)
    if file_path.endswith('tgz'):
        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(path=dir_path)
        tar.close()