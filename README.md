# SingleSliceCardiacStrains
A tool to compute cardiac strains from one slice CINE data and comparing them with strains computed from one- and two-slice DENSE data

## CINE model
The cine model consists of two parts, the main pipeline and the nnU_net model. The main pipeline consists of two files main_cine_strain.py and strain.py. A demo of this model can be found on our [website](https://cecs.ucf.edu/CardiacKinematics).

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


## DENSE model
The DENSE model pipeline is located in the DENSE folder and consists of several Python scripts for loading data, processing, and computing cardiac strains from DENSE MRI.

#### Dependencies:
```
numpy
pandas
scipy
matplotlib
seaborn
```

#### Expected folder structure:
```
DENSE/
├── automatic_implementation.py
├── pipeline/
│   ├── def_grad_tensor.py
│   ├── mat_loader.py
│   ├── rbf_interpolator.py
│   └── transmural_wall_finder.py
├── data/      # Should be added in this style (multiple subjects as example )
│   └── mat/
│   │   ├── Vol_004/ ==> for example, if Vol_id=4 and slice_list=[5,6], it will look for:
│   │   │        ├──  subject004_slice05.mat
│   │   │        └──  subject004_slice06.mat
│   │   ├── Vol_011/
│   │   └── Vol_022/
│   │
│   └── nifti/  ########## OPTIONAL (area_mask_based=True) ########### (in case you want frame area to be computed from nifti masks directly) 
│       ├── Vol_004/ ==> for example, if Vol_id=4 and slice_list=[5,6], it will look for:
│       │        ├──  subject004_slice05.nii.gz    # Each nifti mask has to be saved as an ndarray of [width, height, frame] inside a nifti file
│       │        └──  subject004_slice06.nii.gz
│       ├── Vol_011/
│       └── Vol_022/
├── results_pipeline/
```

### Running the script
With data organized, modify the subject_slices dictionary in the automatic_implementation.py and run:
```
python3 automatic_implementation.py
```

### Acknowledgement
This material is based upon work supported by the National Science Foundation under Award No. 2205043 to Luigi E. Perotti (University of Central Florida) and Dazhong Wu (University of Central Florida), and 2205103 to Daniel B. Ennis (Stanford University and Veterans Affairs Palo Alto Health Care System). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
