from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
import numpy as np
import nibabel as nib


def load_dataset(path_dir):
    labels = []
    for dirname, _, filenames in os.walk(path_dir):
        for name in filenames:
            if name.endswith('.nii'):
                img_path = os.path.join(dirname, name)
                labels.append(img_path)

    inputs = [filename.replace('Mask', 'CT') for filename in labels]

    # if files' name go wrong
    for i in range(len(inputs)):
        if os.path.exists(labels[i]) and os.path.exists(inputs[i]):
            continue
        else:
            print('dont have files'+labels[i])
            exit()

    return inputs, labels


class MyDataset(Dataset):
    def __init__(self,inputs,labels,stack):
        self.inputs, self.labels,self.stack = inputs,labels,stack

    def __getitem__(self, index):
        inputs_nii = nib.load(self.inputs[index])
        inputs_numpy = inputs_nii.get_fdata()

        labels_nii = nib.load(self.labels[index])
        Mask_numpy = labels_nii.get_fdata()


        inputs_min = np.min(inputs_numpy)
        inputs_max = np.max(inputs_numpy)
        CT_normalized = (inputs_numpy - inputs_min) / (inputs_max - inputs_min)

        Mask_min = np.min(Mask_numpy)
        Mask_max = np.max(Mask_numpy)
        Mask_normalized = (Mask_numpy - Mask_min) / (Mask_max - Mask_min)

        # padded_CT
        padded_CT = np.pad(CT_normalized, pad_width=((0, 0), (0, 0), (self.stack, self.stack)), mode='reflect')

        inputs = torch.from_numpy(padded_CT)
        label = torch.from_numpy(Mask_normalized)

        return inputs.permute(2,0,1), label.permute(2,0,1),self.labels[index]

    def __len__(self):
        return len(self.inputs)


def load_data(train=True,stack=3,path_dir = './Mask_train/'):

    if not train:
        path_dir = './Mask_test/'

    inputs, labels = load_dataset(path_dir)
    my_dataset = MyDataset(inputs,labels,stack)

    loader = DataLoader(
        my_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    return loader

