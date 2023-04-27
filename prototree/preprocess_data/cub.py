import os
import shutil
import time
from PIL import Image

path = './data/CUB_200_2011/'

time_start = time.time()

path_images = os.path.join(path, 'images.txt')
path_split = os.path.join(path, 'train_test_split.txt')
train_save_path = os.path.join(path, 'dataset/train_crop/')
test_save_path = os.path.join(path, 'dataset/test_crop/')
bbox_path = os.path.join(path, 'bounding_boxes.txt')

use_segmentation = os.path.isdir(os.path.join(path, 'segmentations'))
print("Using segmentation:", use_segmentation)
train_seg_save_path = os.path.join(path, 'dataset/train_crop_seg/') if use_segmentation else None
test_seg_save_path = os.path.join(path, 'dataset/test_crop_seg/') if use_segmentation else None

images = []
with open(path_images, 'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))
split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split(',')))

bboxes = dict()
with open(bbox_path, 'r') as bf:
    for line in bf:
        id, x, y, w, h = tuple(map(float, line.split(' ')))
        bboxes[int(id)] = (x, y, w, h)

num = len(images)
for k in range(num):
    id, fn = images[k][0].split(' ')
    id = int(id)
    file_name = fn.split('/')[0]
    if int(split[k][0][-1]) == 1:
        dst_dir = train_save_path
        dst_seg_dir = train_seg_save_path
    else:
        dst_dir = test_save_path
        dst_seg_dir = test_seg_save_path

    if not os.path.isdir(os.path.join(dst_dir, file_name)):
        os.makedirs(os.path.join(dst_dir, file_name))
    if use_segmentation and not os.path.isdir(os.path.join(dst_seg_dir, file_name)):
        os.makedirs(os.path.join(dst_seg_dir, file_name))
    img = Image.open(os.path.join(os.path.join(path, 'images'), images[k][0].split(' ')[1])).convert('RGB')
    x, y, w, h = bboxes[id]
    cropped_img = img.crop((x, y, x + w, y + h))
    cropped_img.save(
        os.path.join(os.path.join(dst_dir, file_name), images[k][0].split(' ')[1].split('/')[1]))
    if use_segmentation:
        seg_path = os.path.splitext(images[k][0].split(' ')[1])[0]
        seg_path = os.path.join(path, 'segmentations', seg_path + '.png')
        seg_img = Image.open(seg_path).convert('RGB')
        cropped_img = seg_img.crop((x, y, x + w, y + h))
        cropped_img.save(
            os.path.join(os.path.join(dst_seg_dir, file_name), images[k][0].split(' ')[1].split('/')[1]))
    print('%s' % images[k][0].split(' ')[1].split('/')[1])

train_full_save_path = os.path.join(path, 'dataset/train_full/')
train_seg_save_path = os.path.join(path, 'dataset/train_full_seg/') if use_segmentation else None
train_save_path = os.path.join(path, 'dataset/train_corners/')
train_seg_corners_save_path = os.path.join(path, 'dataset/train_corners_seg/')
test_save_path = os.path.join(path, 'dataset/test_full/')
test_seg_save_path = os.path.join(path, 'dataset/test_full_seg/') if use_segmentation else None

num = len(images)
for k in range(num):
    id, fn = images[k][0].split(' ')
    id = int(id)
    file_name = fn.split('/')[0]
    if int(split[k][0][-1]) == 1:
        if not os.path.isdir(train_full_save_path + file_name):
            os.makedirs(os.path.join(train_full_save_path, file_name))
        if use_segmentation and not os.path.isdir(os.path.join(train_seg_save_path, file_name)):
            os.makedirs(os.path.join(train_seg_save_path, file_name))
        shutil.copy(path + 'images/' + images[k][0].split(' ')[1],
                    os.path.join(
                        os.path.join(train_full_save_path, file_name),
                        images[k][0].split(' ')[1].split('/')[1]))
        if use_segmentation:
            seg_fname = os.path.splitext(images[k][0].split(' ')[1])[0] + '.png'
            shutil.copy(os.path.join(path, 'segmentations', seg_fname),
                        os.path.join(train_seg_save_path, file_name, seg_fname.split('/')[1]))
        if not os.path.isdir(train_save_path + file_name):
            os.makedirs(os.path.join(train_save_path, file_name))
        if use_segmentation and not os.path.isdir(train_seg_corners_save_path + file_name):
            os.makedirs(os.path.join(train_seg_corners_save_path, file_name))

        if use_segmentation:
            seg_fname = os.path.splitext(images[k][0].split(' ')[1])[0] + '.png'
            os.path.join(path, 'segmentations', seg_fname)
        def corners_img(img_path, dir_path, suffix):
            img = Image.open(img_path).convert('RGB')
            x, y, w, h = bboxes[id]
            width, height = img.size
            hmargin = int(0.1 * h)
            wmargin = int(0.1 * w)

            cropped_img = img.crop((0, 0, min(x + w + wmargin, width), min(y + h + hmargin, height)))
            cropped_img.save(os.path.join(dir_path, file_name,"upperleft_" + suffix))
            cropped_img = img.crop((0, max(y - hmargin, 0), min(x + w + wmargin, width), height))
            cropped_img.save(os.path.join(dir_path, file_name,"lowerleft_" + suffix))
            cropped_img = img.crop((max(x - wmargin, 0), 0, width, min(y + h + hmargin, height)))
            cropped_img.save(os.path.join(dir_path, file_name, "upperright_" + suffix))
            cropped_img = img.crop(((max(x - wmargin, 0), max(y - hmargin, 0), width, height)))
            cropped_img.save(os.path.join(dir_path, file_name, "lowerright_" + suffix))
            img.save(os.path.join(dir_path, file_name,"normal_" + suffix))

        img_path = os.path.join(os.path.join(path, 'images'), images[k][0].split(' ')[1])
        suffix = images[k][0].split(' ')[1].split('/')[1]
        corners_img(img_path, train_save_path, suffix)
        if use_segmentation:
            seg_fname = os.path.splitext(images[k][0].split(' ')[1])[0] + '.png'
            img_path = os.path.join(path, 'segmentations', seg_fname)
            corners_img(img_path, train_seg_corners_save_path, suffix)

        print('%s' % images[k][0].split(' ')[1].split('/')[1])
    else:
        if not os.path.isdir(os.path.join(test_save_path, file_name)):
            os.makedirs(os.path.join(test_save_path, file_name))
        if use_segmentation and not os.path.isdir(os.path.join(test_seg_save_path, file_name)):
            os.makedirs(os.path.join(test_seg_save_path, file_name))
        shutil.copy(path + 'images/' + images[k][0].split(' ')[1],
                    os.path.join(test_save_path, file_name, images[k][0].split(' ')[1].split('/')[1]))
        if use_segmentation:
            seg_fname = os.path.splitext(images[k][0].split(' ')[1])[0] + '.png'
            shutil.copy(os.path.join(path, 'segmentations', seg_fname),
                        os.path.join(test_seg_save_path, file_name, seg_fname.split('/')[1]))
        print('%s' % images[k][0].split(' ')[1].split('/')[1])
time_end = time.time()
print('CUB200, %s!' % (time_end - time_start))
