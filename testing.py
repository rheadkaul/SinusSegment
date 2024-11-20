from load_dataset import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from UnetPlusPlus import UnetPlusPlus
import math
from criterion import compute_iou,compute_dice,compute_sensitivity_specificity,hausdorff_distance_for_segmentation
import numpy as np
import cv2 
import nibabel as nib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

c_in, c_out, stack = 1, 1, 3
model = UnetPlusPlus(stack = stack*2+1,num_classes=1, deep_supervision=True).to(device=device, dtype=dtype)

checkpoint_path = './parameters/default_parameters.pth'
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state'])
    print('loading parameters successfully')
except:
    print('could not load parameters')
    exit()


train_loader = load_data(False,stack)

batch_size = 4

avg_iou = 0.0
avg_dice = 0.0
avg_sen = 0.0
avg_sep = 0.0

total_index = 0

image_index = 10000
for k, (inputs, labels, name) in tqdm(enumerate(train_loader, 1)):
    patient_iou = 0.0
    patient_dice = 0.0
    patient_sen = 0.0
    patient_sep = 0.0
    avg_hausdorff_distance = []
    segement_list =[]
    new_name = name[0].split('/')[-1]

    labels = labels.permute(1, 0, 2, 3)
    inputs = inputs.squeeze(0)

    # sliding widow
    window_size = stack * 2 + 1
    stride = 1
    unfolded_input = inputs.unfold(0, window_size, stride)
    unfolded_input = unfolded_input.permute(0, 3, 1, 2)

    num_batches = math.ceil(unfolded_input.shape[0] / batch_size)

    index = 0
    instance_loss = 0
    for i in range(num_batches):
        inputs_batch = unfolded_input[i * batch_size: (i + 1) * batch_size]  # 每个batch取4个
        labels_batch = labels[i * batch_size: (i + 1) * batch_size]


        if len(labels_batch.shape) == 3:
            labels_batch = labels_batch.unsqueeze(0)

        inputs_batch, labels_batch = inputs_batch.to(device=device, dtype=dtype), labels_batch.to(device=device,dtype=dtype)

        with torch.no_grad():
            output_list = model(inputs_batch)
        # get the last prediction
        output = output_list[3]

        if labels_batch.sum(dim=(1,2,3)) !=0:

          patient_iou += compute_iou(output,labels_batch)
          patient_dice += compute_dice(output, labels_batch)
          sen,sep = compute_sensitivity_specificity(output,labels_batch)

          if output.sum(dim=(1,2,3)) !=0:
            HD =  hausdorff_distance_for_segmentation(output, labels_batch)
            if HD != 0.0:
                avg_hausdorff_distance.append(HD)

          patient_sen += sen
          patient_sep += sep
        
          index +=1
          total_index +=1
          
        
        labels_batch = (labels_batch * 255.0).clamp(0, 255).to(torch.uint8)
        output = (output * 255.0).clamp(0, 255).to(torch.uint8)
        # b c h w -> b h w c
        labels_batch = labels_batch.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        output = output.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        output = output[0,:,:,0]
        segement_list.append(output)

    # saving prediction
    segment_array = np.array(segement_list).transpose(1, 2, 0)
    # load original nii
    roi_img = nib.load(name[0])
    # get affine
    roi_dat = roi_img.get_fdata()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_img = nib.Nifti1Image(segment_array, roi_aff, header=roi_hdr)
    new_path = './seg_prediction/' + new_name
    nib.save(opt_img, new_path)

    print('patient name: {}, iou: {}, dice: {}, sen: {}, sep: {}, hausdorff: {}'.format(new_name,patient_iou/index,patient_dice/index,patient_sen/index,patient_sep/index,sum(avg_hausdorff_distance) / len(avg_hausdorff_distance)))
    avg_iou += patient_iou/index
    avg_dice += patient_dice/index
    avg_sen += patient_sen/index
    avg_sep += patient_sep/index

print('all testing dataset avg_iou: {}, avg_dice: {}, avg_sen: {}, avg_sep: {}'.format(avg_iou/k,avg_dice/k,avg_sen/k,avg_sep/k))










