import os
import pydicom
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from strain import compute_strain_cine
import glob
from pathlib import Path
from skimage import measure


def remove_islands(mask):
    """
    Remove islands from prediction
    
    parameters:
        mask (2D array): prediction mask
    
    returns:
        clean_mask (2D array): mask without islands
    """
    # Find all chunks of pixels
    labels = measure.label(mask, connectivity=2)
    
    # If there is only one chunk, no islands on original mask
    if labels.max() == 0:
        return mask
    
    # Find the largest chunk
    largest_cc = np.argmax(np.bincount(labels.flat)[1:]) + 1
    
    # Remove all the other ones
    clean_mask = (labels == largest_cc).astype(np.uint8)
    
    # Return cleaned mask
    return clean_mask

MODEL_LOC   = ''    # Location of the nnunet model
OUTPUT_LOC  = ''    # Result location
INPUT_LOC   = ''    # Location of the input files

predictor = nnUNetPredictor()
predictor.initialize_from_trained_model_folder(MODEL_LOC, use_folds=(0,1,2,3,4), checkpoint_name="checkpoint_final.pth")


# Loop over slices
for f in glob.glob(os.path.join(INPUT_LOC, '*')):
    original_file_name = f.split("/")[-1]
    result_local_location = os.path.join(OUTPUT_LOC, original_file_name)
    os.makedirs(result_local_location, exist_ok=True)

    images = []  # Store image arrays here
    input_images = [] # Input image are stored in here
    pred_array = [] # Store prediction for every frame in here

    for dcm in sorted(glob.glob(os.path.join(f, '*.dcm')), key=lambda p: Path(p).name):
        ds = pydicom.dcmread(dcm)
        pixel_space = ds.PixelSpacing
        img = ds.pixel_array
        input_images.append(img)

    ####  nnUNet implementation  ####
        # Set up config for specific image, so the model understand what it has to do
        img = np.asarray(img)
        slice_input = img[None, None, :, :] # This is necessary because the model needs a 4D array even if it was trained as 2D
        
        # Set properties of the nnunet
        properties = {
            'spacing': [pixel_space[0], pixel_space[1],1.0],  # Change this to be based on the user's image spacing
            'original_size_of_raw_data': img.shape,
            'original_shape_of_raw_data': img.shape,
        }

        # Run prediction
        pred = predictor.predict_single_npy_array(
            slice_input,                     # shape [1,1,H,W]
            properties,
        )
        
        # Get the prediction mask only
        pred = pred[0]
        
        # Remove cavity
        idx_lvc = pred == 2
        pred[idx_lvc] = 0

        pred = remove_islands(pred)
        pred_array.append(pred)

    # Compute strain
    err, ell, epi, endo = compute_strain_cine(pred_array)

    # Save segmentation
    pred_array = np.array(pred_array)
    np.savez(os.path.join(result_local_location, 'segmentation_masks.npz'), pred_array)

    # Txt file / write / print results
    txt_file_name = 'results.txt'
    with open(os.path.join(result_local_location, txt_file_name), 'w', encoding='utf-8') as file:
        file.write("Strain Results\n")
        for c_f in range(len(err)):
            err_str = "Err at t " + str(c_f) + ": " + str(err[c_f])
            file.write(err_str + '\n')
            print(err_str)

        file.write('\n')

        for c_f in range(len(ell)):
            ell_str = "Ell at t " + str(c_f) + ": " + str(ell[c_f])
            file.write(ell_str + '\n')
            print(ell_str)

        file.write('\n')

        for c_f in range(len(endo)):
            endo_str = "Ecc_endo at t " + str(c_f) + ": " + str(endo[c_f])
            file.write(endo_str + '\n')
            print(endo_str)

        file.write('\n')

        for c_f in range(len(epi)):
            epi_str = "Ecc_epi at t " + str(c_f) + ": " + str(epi[c_f])
            file.write(epi_str + '\n')
            print(epi_str)

        file.close()
