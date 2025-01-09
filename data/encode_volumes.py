import argparse
import os
import gc
import torch
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import SimpleITK as sitk

from glob import glob
from tqdm import tqdm

import models_vit


def load_model(vit_model, ckpt_path):

    # Build ViT model without classification head
    model = models_vit.__dict__[vit_model](
        num_classes=0,
        global_pool=False,
        in_chans=1,
        img_size=512,
    )

    # Load model checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    return model


def clip_and_normalize(np_image: np.ndarray,
                       clip_min: int = -150,
                       clip_max: int = 250
                       ) -> np.ndarray:
    np_image = np.clip(np_image, clip_min, clip_max)
    np_image = (np_image - clip_min) / (clip_max - clip_min)
    return np_image


resize = transforms.Resize((512, 512))


def main(images, args):

    model = load_model(args.vit_model, args.ckpt_path)
    model.eval()
    model.to('cuda')

    for image_file in tqdm(images):

        image = sitk.ReadImage(image_file)
        image = sitk.GetArrayFromImage(image)
        image = clip_and_normalize(image)

        tensor = torch.from_numpy(image.astype('float32'))
        tensor = tensor.unsqueeze(1)
        tensor = resize(tensor)
        tensor = tensor.to('cuda')

        outputs = []
        for i in tqdm(range(0, tensor.size(0), args.batch_size)):
            batch = tensor[i:i + args.batch_size]
            with torch.no_grad():  # Ensure no gradients are computed to save memory
                output = model(batch)
            outputs.append(output)

        # Concatenate all batch outputs
        outputs = torch.cat(outputs, dim=0)
        outputs_np = outputs.cpu().numpy()

        # Clip values outside the 5th and 95th percentiles
        p5 = np.percentile(outputs_np, 5)
        p95 = np.percentile(outputs_np, 95)
        outputs_np = np.clip(outputs_np, p5, p95)

        # Save to disk
        np.save(os.path.join(args.save_folder, os.path.basename(image_file).replace('nii.gz', 'npy')), outputs_np)

        # Clear GPU memory
        del tensor, outputs, outputs_np  # Delete unused tensors
        gc.collect()  # Garbage collector
        torch.cuda.empty_cache()  # Clear cache


if __name__ == '__main__':

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a part of a DataFrame.")
    parser.add_argument("--part_num", type=int, default=1, help="The part number to process (1-indexed).")
    parser.add_argument("--total_parts", type=int, default=12, help="The total number of parts to divide the DataFrame into.")
    parser.add_argument("--data_root", type=str, default='/data/houbb/data/CT-RATE/dataset/')
    parser.add_argument("--metadata_csv", type=str, default='radiology_text_reports/validation_reports.csv')
    parser.add_argument("--split", type=str, default='valid')
    parser.add_argument("--save_folder", type=str, default='./features')
    parser.add_argument("--ckpt_path", type=str, default='/data/houbb/scratch/tcia_mae/41813900/checkpoint-260.pth')
    parser.add_argument("--vit_model", type=str, default='vit_g_patch16')
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Glob all nifti volumes
    metadata_df = pd.read_csv(os.path.join(args.data_root, args.metadata_csv))
    metadata_df = metadata_df.drop_duplicates(subset='Findings_EN')
    metadata_df = metadata_df.dropna(subset='Findings_EN')

    def make_filepath(VolumeName):
        dir1 = VolumeName.rsplit('_', 1)[0]
        dir2 = VolumeName.rsplit('_', 2)[0]
        return os.path.join(args.data_root, args.split, dir2, dir1, VolumeName)

    images = metadata_df['VolumeName'].apply(make_filepath).values

    # Calculate the number of rows in each part
    total_images = len(images)
    part_size = total_images // args.total_parts
    remainder = total_images % args.total_parts

    # Calculate the start and end indices for the slice
    start = (args.part_num - 1) * part_size + min(args.part_num - 1, remainder)
    end = start + part_size + (1 if args.part_num <= remainder else 0)

    main(images[start:end], args)
