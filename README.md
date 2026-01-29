# SingleSliceCardiacStrains
Computing cardiac strains from one slice CINE data and comparing it with strains computed from one- and two-slice DENSE data

## CINE model
The cine model consists of two parts, the main pipeline and the nnU_net model. The main pipeline consists of two files main_cine_strain.py and strain.py.

### cine_model_nnUnet.zip
This file is the trained model used in this work. All folds, weights, and required files are include in this file.

### Strain.py
This is the main function/pipeline for this work, it is responsible for taking all measurements and computing longitudinal (Ell), circumferential (Ecc), and radial strain. This file has only functions, therefore, no changes are required.

### main_cine_strain.py
This is the main file where the strain.py is called. This file is responsible for running the nnU_net model directly in Python and display the results. 

#### Dependencies:
```
numpy
nnunetv2
scikit-image
scipy
pydicom
```

#### Path variables (requires manual assignment before usage):
```
MODEL_LOC   -> location of the nnU_net model
OUTPUT_LOC  -> Folder that will be created to save the result files
INPUT_LOC   -> Location of the input slice
```

#### Expected folder structure:
```
Data_folder
├── Slice1
    ├── Frame1.dcm
    ├── Frame2.dcm
    ├── Frame3.dcm
├── Slice2
    ├── Frame1.dcm
    ├── Frame2.dcm
    ├── Frame3.dcm
```

### Running the script
With data organized and the path variables assigned, run:
```
python3 main_cine_strain.py
```

