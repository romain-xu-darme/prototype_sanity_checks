import torch
from settings import *
from prp import *
from preprocess import *
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-img', nargs=1, type=str)
parser.add_argument('-proto', nargs=1, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

test_image_dir = args.imgdir[0]
test_image_name = args.img[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]
prototype_number = args.proto[0]

### Load trained ProtoPNet model
load_model_path = os.path.join(load_model_dir, load_model_name)
ppnet = torch.load(load_model_path,map_location=torch.device(device))
prp_model = PRPCanonizedModel(ppnet,base_architecture)

img_size = ppnet.img_size


### For a single test image and for a particular prototype
test_image_path = os.path.join(test_image_dir, test_image_name)
normalize = transforms.Normalize(mean=mean, std=std)
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize
])
img_pil = Image.open(test_image_path)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
images_test = img_variable.cuda()
print(images_test[0,0,:3,:3])
prp_map = generate_prp_image(images_test, prototype_number, prp_model, device)
makedir("Test images/PRP/")
plt.imsave("Test images/PRP/"+"prp_"+str(prototype_number)+"_"+test_image_name, prp_map, cmap="seismic", vmin=-1, vmax=+1)



#############
### OVERLAY ##########
################
def invert_normalize(ten, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    s = torch.tensor(np.asarray(std, dtype=np.float32)).unsqueeze(1).unsqueeze(2)
    m = torch.tensor(np.asarray(mean, dtype=np.float32)).unsqueeze(1).unsqueeze(2)

    res = ten * s + m
    return res

heatmap = cv2.imread("Test images/PRP/"+"prp_"+str(prototype_number)+"_"+test_image_name)
heatmap = heatmap[..., ::-1]
heatmap = np.float32(heatmap) / 255
ts = invert_normalize(img_tensor.squeeze())
a = ts.data.numpy().transpose((1, 2, 0))
overlayed_original_img_j = 0.2 * a + 0.6 * heatmap
plt.imsave("Test images/PRP/"+"Overlay_prp_"+str(prototype_number)+"_"+test_image_name,
           overlayed_original_img_j,
           vmin=-1,
           vmax=+1.0)



