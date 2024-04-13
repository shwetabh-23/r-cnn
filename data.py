import json
import pandas as pd
import os
from torch.utils.data import Dataset
import PIL
import numpy as np
import ast

import torchvision.transforms as transforms

def denormalize_bbox(normalized_bbox, image_width, image_height):
    
    # Unpack normalized coordinates
    x, y, w, h = normalized_bbox
    # Convert to absolute coordinates
    x_min = int((float(x) - float(w) / 2) * image_width)
    y_min = int((float(y) - float(h) / 2) * image_height)
    x_max = int((float(x) + float(w) / 2) * image_width)
    y_max = int((float(y) + float(h) / 2) * image_height)
    
    return x_min, y_min, x_max, y_max

def generate_csv_files(root_file_path, type):

    if type == 'coco':
        with open(root_file_path, 'r') as json_file:
            file = json.load(json_file)

        annotations_csv = pd.DataFrame(file['annotations'])
        image_csv = pd.DataFrame(file['images'])
        result = pd.merge(annotations_csv, image_csv, left_on= 'image_id', right_on= 'id', how='inner')
        result.drop(['id_x', 'id_y', 'image_id', 'iscrowd', 'area', 'segmentation', 'depth', 'height', 'width'], axis= 1, inplace= True)
        return result

    if type == 'yolo':
        lines = []
        imgs = []
        for file_path in os.listdir(root_file_path):
            
            file = os.path.join(root_file_path, file_path)
            try:
                img = PIL.Image.open(os.path.join(r'data/yolo_1k/train/images', file_path[:-4]+'.png'))
            except:
                img = PIL.Image.open(os.path.join(r'data/yolo_1k/val/images', file_path[:-4]+'.png'))
            with open(file, 'r') as txt:
                lines_split = [line.strip().split() for line in txt]
                lines.extend(lines_split)
                imgs.append(file_path[:-4]+'.png')
        
        df = pd.DataFrame(lines, imgs)
        df.columns = ['category_id', 'bbox_1', 'bbox_2', 'bbox_3', 'bbox_4']
        df['bbox'] = df[['bbox_1', 'bbox_2', 'bbox_3', 'bbox_4']].apply(lambda x : x.tolist(), axis= 1)
        df.drop(['bbox_1', 'bbox_2', 'bbox_3', 'bbox_4'], axis = 1, inplace= True)
        df['bbox'] = df['bbox'].apply(lambda x : denormalize_bbox(x, img.size[0], img.size[1]))
        
        return df
    
class dataset(Dataset):
    def __init__(self, csv_file, image_size, type, root_img_path):

        self.csv_file = pd.read_csv(csv_file)
        self.image_size = image_size
        self.type = type
        self.root_img_path = root_img_path

    def transform(self, img):
        transforms_fn = transforms.Compose([transforms.Resize(self.image_size),
              # Resize the image to (500, 500)
            transforms.ToTensor(), transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])           # Convert the image to a PyTorch tensor
        ])
        tensor_image = transforms_fn(img)
        return tensor_image

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):

    
        image_path = os.path.join(self.root_img_path, self.csv_file['file_name'][idx])
        orig_img = PIL.Image.open(image_path)
        img = self.transform(orig_img)

        bbox = self.csv_file['bbox'][idx]
        bbox = np.array(ast.literal_eval(bbox))
        x, y, width, height = bbox
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        width_scale_factor = self.image_size[1] / orig_img.size[0] 
        height_scale_factor = self.image_size[0] / orig_img.size[1]
        x1, x2 = x1 * width_scale_factor, x2 * width_scale_factor 
        y1, y2 = y1 * height_scale_factor, y2 * height_scale_factor 
        bbox = [x1, y1, x2, y2]
        bbox = np.expand_dims(bbox, axis=1)

        bbox = np.transpose(bbox)
        category = self.csv_file['category_id'][idx]

        return img, bbox, category