from PIL import Image
from matplotlib import pyplot as plt
import json
from data import generate_csv_files, dataset
import pandas as pd
import os 
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import draw_boxes_on_image, gen_anc_centers, display_img, display_grid, gen_anc_base, project_bboxes, display_bbox, get_req_anchors
import torchvision
import torch
from model import TwoStageDetector
from tqdm import tqdm
# Load the PNG image
# image_path = r"data/coco_1k/train2017/Calc-Training_P_00022_LEFT_MLO.png"
# image = Image.open(image_path)

# # Display the image
# plt.imshow(image)
# plt.axis('off')  # Turn off axis
# plt.show()

# train_coco = generate_csv_files(root_file_path= r'data/coco_1k/annotations/instances_train2017.json', type= 'coco')
# val_coco = generate_csv_files(root_file_path= r'data/coco_1k/annotations/instances_val2017.json', type= 'coco')
# train_coco.to_csv('train_coco.csv')
# val_coco.to_csv('val_coco.csv')
# train_yolo = generate_csv_files(root_file_path=r'data/yolo_1k/train/labels', type= 'yolo')
# val_yolo = generate_csv_files(root_file_path=r'data/yolo_1k/val/labels', type= 'yolo')

data_train = dataset(csv_file= r'train_coco.csv', image_size= (256, 256), type= 'coco', root_img_path= r'data/coco_1k/train2017')
data_val = dataset(csv_file= r'val_coco.csv', image_size= (256, 256), type= 'coco', root_img_path= r'data/coco_1k/val2017')

train_loader = DataLoader(data_train, batch_size= 32)
val_loader = DataLoader(data_val, batch_size= 32)

model = torchvision.models.resnet50(pretrained=True)
req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers)
for param in backbone.named_parameters():
    param[1].requires_grad = True
img_data_all, bbox, classes = (next(iter(train_loader)))

img_width, img_height = img_data_all.shape[2], img_data_all.shape[3]
out = backbone(img_data_all)
out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)
print(out_c, out_h, out_w)

width_scale_factor = img_width / out_w
height_scale_factor = img_height / out_h
anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(out_h, out_w))

anc_pts_x_proj = anc_pts_x.clone() * width_scale_factor 
anc_pts_y_proj = anc_pts_y.clone() * height_scale_factor

nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
fig, axes = display_img(img_data_all[:2], fig, axes)
fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])

anc_scales = [2, 4, 6]
anc_ratios = [0.5, 1, 1.5]
n_anc_boxes = len(anc_scales) * len(anc_ratios) # number of anchor boxes for each anchor point
anc_base = gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, (out_h, out_w))
anc_boxes_all = anc_base.repeat(img_data_all.size(dim=0), 1, 1, 1, 1)
# nrows, ncols = (1, 2)
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

# fig, axes = display_img(img_data_all[:2], fig, axes)

# # project anchor boxes to the image
# anc_boxes_proj = project_bboxes(anc_boxes_all, width_scale_factor, height_scale_factor, mode='a2p')

# # plot anchor boxes around selected anchor points
# sp_1 = [5, 8]
# sp_2 = [12, 9]
# bboxes_1 = anc_boxes_proj[0][sp_1[0], sp_1[1]]
# bboxes_2 = anc_boxes_proj[1][sp_2[0], sp_2[1]]

# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0], (anc_pts_x_proj[sp_1[0]], anc_pts_y_proj[sp_1[1]]))
# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1], (anc_pts_x_proj[sp_2[0]], anc_pts_y_proj[sp_2[1]]))
# fig, _ = display_bbox(bboxes_1, fig, axes[0])
# fig, _ = display_bbox(bboxes_2, fig, axes[1])

# nrows, ncols = (1, 4)
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

# fig, axes = display_img(img_data_all[:4], fig, axes)

# # plot feature grid
# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])

# # plot all anchor boxes
# for x in range(anc_pts_x_proj.size(dim=0)):
#     for y in range(anc_pts_y_proj.size(dim=0)):
#         bboxes = anc_boxes_proj[0][x, y]
#         fig, _ = display_bbox(bboxes, fig, axes[0], line_width=1)
#         fig, _ = display_bbox(bboxes, fig, axes[1], line_width=1)
pos_thresh = 0.7
neg_thresh = 0.3

# project gt bboxes onto the feature map
gt_bboxes_proj = project_bboxes(bbox, width_scale_factor, height_scale_factor, mode='p2a')
positive_anc_ind, negative_anc_ind, GT_conf_scores, \
GT_offsets, GT_class_pos, positive_anc_coords, \
negative_anc_coords, positive_anc_ind_sep = get_req_anchors(anc_boxes_all, gt_bboxes_proj, classes, pos_thresh, neg_thresh)

pos_anc_proj = project_bboxes(positive_anc_coords, width_scale_factor, height_scale_factor, mode='a2p')
neg_anc_proj = project_bboxes(negative_anc_coords, width_scale_factor, height_scale_factor, mode='a2p')

# grab +ve and -ve anchors for each image separately

anc_idx_1 = torch.where(positive_anc_ind_sep == 0)[0]
anc_idx_2 = torch.where(positive_anc_ind_sep == 1)[0]

pos_anc_1 = pos_anc_proj[anc_idx_1]
pos_anc_2 = pos_anc_proj[anc_idx_2]

neg_anc_1 = neg_anc_proj[anc_idx_1]
neg_anc_2 = neg_anc_proj[anc_idx_2]

# nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

fig, axes = display_img(img_data_all[:2], fig, axes)

# # plot groundtruth bboxes
fig, _ = display_bbox(bbox[0], fig, axes[0])
fig, _ = display_bbox(bbox[1], fig, axes[1])

# plot positive anchor boxes
fig, _ = display_bbox(pos_anc_1, fig, axes[0], color='g')
fig, _ = display_bbox(pos_anc_2, fig, axes[1], color='g')

# plot negative anchor boxes
fig, _ = display_bbox(neg_anc_1, fig, axes[0], color='r')
fig, _ = display_bbox(neg_anc_2, fig, axes[1], color='r')
img_size = (img_height, img_width)
out_size = (out_h, out_w)
n_classes =  1 # exclude pad idx
roi_size = (2, 2)

detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)

# detector.eval()
# total_loss = detector(img_data_all, bbox, classes)
# proposals_final, conf_scores_final, classes_final = detector.inference(img_data_all)
# breakpoint()

def training_loop(model, learning_rate, train_dataloader, n_epochs, device):
    # Move model to the specified device (CPU or GPU)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
            # Move batch data to the specified device
            img_batch = img_batch.to(device)
            gt_bboxes_batch = gt_bboxes_batch.cuda()
            gt_classes_batch = gt_classes_batch.cuda()
            
            # Forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        loss_list.append(total_loss)
        
    return loss_list

learning_rate = 1e-3
n_epochs = 1000

loss_list = training_loop(detector, learning_rate, train_loader, n_epochs, 'cuda')
breakpoint()
#fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])
# for i in range(50, 60):
#     data = data_train.__getitem__(i)
#     img, box, xlass = data

#     draw_boxes_on_image(img, box)
