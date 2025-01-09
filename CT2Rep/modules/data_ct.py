import os
import glob
import json
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]


class CTReportDataset(Dataset):
    def __init__(self, args, data_folder, csv_file, tokenizer, min_slices=20, resize_dim=500, num_frames=2, force_num_frames=True):
        self.min_slices = min_slices
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        metadata_df = pd.read_csv(csv_file)
        metadata_df = metadata_df.drop_duplicates(subset='Findings_EN')
        metadata_df = metadata_df.dropna(subset='Findings_EN')

        def make_filepath(VolumeName):
            dir1 = VolumeName.rsplit('_', 1)[0]
            dir2 = VolumeName.rsplit('_', 2)[0]
            return os.path.join(data_folder, dir2, dir1, VolumeName)
        metadata_df['filepaths'] = metadata_df['VolumeName'].apply(make_filepath)

        metadata_df['input_ids'] = metadata_df['Findings_EN'].apply(
            lambda x: tokenizer(x, max_length=self.max_seq_length)['input_ids'])

        self.metadata_df = metadata_df

        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        # self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.metadata_df)

    def nii_img_to_tensor(self, path, transform):
        img_data = sitk.ReadImage(path)
        img_data = sitk.GetArrayFromImage(img_data)
        img_data= np.transpose(img_data, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)

        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
        
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)
        
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]

    
    def __getitem__(self, index):
        row = self.metadata_df.iloc[index]
        img_id = row['VolumeName']
        tensor = self.nii_to_tensor(row['filepaths'])
        ids = row['input_ids']
        mask = [1] * len(ids)
        seq_length = len(ids)
        sample = (img_id, tensor, ids, mask, seq_length)
        return sample



if __name__ == '__main__':

    a=1

    from transformers import AutoTokenizer

    dataset = CTReportDataset(
        args=None,
        data_folder='/data/houbb/data/CT-RATE/dataset/valid_fixed',
        csv_file='/data/houbb/data/CT-RATE/dataset/radiology_text_reports/validation_reports.csv',
        tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
    )

    a=1

    sample = dataset[42]

    a=1
