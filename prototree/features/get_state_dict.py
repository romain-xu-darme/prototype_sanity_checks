import gdown
import os

dir_path = 'prototree/features/state_dicts'
os.makedirs(dir_path, exist_ok=True)
url = "https://drive.google.com/uc?id=15n8iP17lpr61GQp8pW0tGdYaU7TJqNRS"
gdown.download(url, os.path.join(dir_path,'BBN.iNaturalist2017.res50.180epoch.best_model.pth'), quiet=False)
