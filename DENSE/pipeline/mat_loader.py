


import os
import numpy as np
import scipy.io as sio

def polygon_area(coords):
    """
    Calculate the signed area of a polygon using the Shoelace formula.
    Returns absolute area.
    """
    if coords is None or coords.size == 0:
        return 0.0
    # Ensure coords is (N, 2)
    if coords.ndim != 2 or coords.shape[1] != 2:
        return 0.0
        
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def data_loader(Vol_id, slice_list, data_root="data/mat"):
    """
    This function loads displacement and coordinate data from .mat files for the specified volume and slices.

    Parameters:
    - Vol_id: Volume ID (integer)
    - slice_list: List of slice names (integers)
    - data_root: Root directory where the .mat files are stored (default: "data/mat")
       |
       |==> for example, if Vol_id=4 and slice_list=[5,6], it will look for:
       |         data/mat/Vol_004/subject004_slice05.mat
       |         data/mat/Vol_004/subject004_slice06.mat
       |
       |==> for example, if Vol_id=9 and slice_list=[7,8], it will look for:
       |         data/mat/Vol_009/subject009_slice07.mat
       |         data/mat/Vol_009/subject009_slice08.mat
           

    Returns:
    - lagrangian_displacement_mm: Numpy array of shape (p, 3, t) containing Lagrangian displacements in mm
    - coordinates_mm: Numpy array of shape (p, dimension) containing coordinates in mm
    - endo_contours_mm: List of endocardial contours for each slice in mm
    - epi_contours_mm: List of epicardial contours for each slice in mm
    - voxel_sizes: Tuple containing voxel sizes in mm (voxel_size_x, voxel_size_y, voxel_size_z)
    - lv_area_slices: List of lists containing LV areas for each frame in each slice
    - peak_systole_frame_volume: Integer indicating the peak systole frame for the entire volume
    - first_slice_points: Number of points in the first slice (integer) - Useful for indexing when multiple slices are present
    """
    # sort slice list and check consecutiveness
    slice_list = sorted(slice_list)
    if len(slice_list) > 1:
        # Check if differences between adjacent elements are all 1
        if not all(slice_list[i] == slice_list[i-1] + 1 for i in range(1, len(slice_list))):
            raise ValueError(f"Provided slices are not consecutive: {slice_list}")

    # prepare volume directory path
    vol_str = f"{int(Vol_id):03d}"
    vol_dir = os.path.join(data_root, f"Vol_{vol_str}")
    full_depth = len(slice_list)

    # Determine minimum number of frames across slices
    min_num_frames = float('inf')
    number_of_points_all_slices = 0
    for slice_name in slice_list:
        slice_str = f"{int(slice_name):02d}"
        filename = f"subject{vol_str}_slice{slice_str}.mat"
        filepath = os.path.join(vol_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        mat = sio.loadmat(filepath)
        disp_info = mat['DisplacementInfo']
        dX_sample = disp_info['dX'][0, 0] 
        
        num_frames = dX_sample.shape[1]
        min_num_frames = min(min_num_frames, num_frames)
        number_of_points_slice = dX_sample.shape[0] 
        number_of_points_all_slices += number_of_points_slice


    # Initialize containers
    slice_thickness = mat['SequenceInfo'][0, 0]['SliceThickness'][0]
    voxel_size_x, voxel_size_y = mat['SequenceInfo'][0, 0]['PixelSpacing'][0], mat['SequenceInfo'][0, 0]['PixelSpacing'][1]
    voxel_size_z = voxel_size_x  # Assuming square pixels in-plane for DENSE

    # initialize output arrays
    data_dimension = 3 if len(slice_list) > 1 else 2
    lagrangian_displacement_voxel = np.ndarray(shape=(number_of_points_all_slices, 3, min_num_frames), dtype=float) # (p, 3, t)
    print(f"Initialized lagrangian_displacement_voxel with shape: {lagrangian_displacement_voxel.shape}")
    coordinates_voxel = np.ndarray(shape=(number_of_points_all_slices, data_dimension), dtype=float) # (p, dimension)
    print(f"Initialized coordinates_voxel with shape: {coordinates_voxel.shape}")


    endo_contours = []
    epi_contours = []
    lv_area_slices = [] # To store area per frame for each slice
    slices_peak_systole = []

    barry_centers = []

    for z, slice_name in enumerate(slice_list): # z is depth index : 0 for the first slice, 1 for the second, ...
        slice_str = f"{int(slice_name):02d}"
        filename = f"subject{vol_str}_slice{slice_str}.mat"
        filepath = os.path.join(vol_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        mat = sio.loadmat(filepath)

        # --- Process DisplacementInfo ---
        displacement = mat['DisplacementInfo']
        
        # Unwrap MATLAB structs
        X = np.asarray(displacement['X'][0, 0]).squeeze()
        Y = np.asarray(displacement['Y'][0, 0]).squeeze()
        dX = np.asarray(displacement['dX'][0, 0])
        dY = np.asarray(displacement['dY'][0, 0])
        dZ = np.asarray(displacement['dZ'][0, 0]) # if 'dZ' in displacement.dtype.names else None



        
        ################################### Bring the barry center at 0,0 for all slices

        x_centoid_nonTranslated = np.mean(X)
        y_centoid_nonTranslated = np.mean(Y)

        # Translate X and Y to have centroid at (0,0)
        X = X - x_centoid_nonTranslated
        Y = Y - y_centoid_nonTranslated

        x_centroid_translated = np.mean(X)
        y_centroid_translated = np.mean(Y)
        print(f"The centroid after translation for slice {slice_name} is at ({x_centroid_translated}, {y_centroid_translated})")


        # Normalize shapes
        X = X.flatten()
        Y = Y.flatten()

        ###################################### accumulate the displacement and coordinates in voxel space
        # define cumulative point index based on Z
        if z == 0:
            first_slice_points = X.shape[0] # starting point index for the second slice is number of points in the first slice
    
        
        for p in range(X.shape[0]):
            global_p = p if z == 0 else p + first_slice_points
            coordinates_voxel[global_p, :] = [X[p], Y[p], z] if data_dimension == 3 else [X[p], Y[p]]
            for direction in range(3):
                for f in range(min_num_frames):
                    if direction == 0:
                        lagrangian_displacement_voxel[global_p, direction, f] = dX[p, f]
                    elif direction == 1:
                        lagrangian_displacement_voxel[global_p, direction, f] = dY[p, f]
                    elif direction == 2:
                        lagrangian_displacement_voxel[global_p, direction, f] = dZ[p, f]


        ############################################# --- Extract Contours
        if 'ROIInfo' in mat:
            try:
                roi_info = mat['ROIInfo'][0, 0]
                resting = roi_info['RestingContour']
                
                epi_resting = resting[0, 0]
                endo_resting = resting[0, 1]
                

                ## perform the shift here based on barry centers
                epi_resting_centered = np.zeros_like(epi_resting)
                endo_resting_centered = np.zeros_like(endo_resting)
                # translate x
                epi_resting_centered[:, 0] = epi_resting[:, 0] - x_centoid_nonTranslated
                endo_resting_centered[:, 0] = endo_resting[:, 0] - x_centoid_nonTranslated
                # translate y
                epi_resting_centered[:, 1] = epi_resting[:, 1] - y_centoid_nonTranslated
                endo_resting_centered[:, 1] = endo_resting[:, 1] - y_centoid_nonTranslated


      
            except Exception:
                pass
        
        endo_contours.append(endo_resting_centered) # the list of endo contours (resting) for all slices
        epi_contours.append(epi_resting_centered) # the list of epi contours (resting) for all slices

    #   ############################### --- Extract Contours and Calculate Areas ---
        lv_area_frames = [] # To store area per frame for this slice]
        min_lvc_area = float('inf')
        peak_systole_frame = 0
        if 'ROIInfo' in mat:
            try:
                roi_info = mat['ROIInfo'][0, 0]
                # Check for 'Contour' field which contains dynamic contours
                if 'Contour' in roi_info.dtype.names:
                    contours_cell = roi_info['Contour']
                    
                    if contours_cell.shape[0] == min_num_frames:
                        for f in range(min_num_frames):
                            epi_frame = contours_cell[f, 0]     # no shift needed here as it does not affect area calculation
                            endo_frame = contours_cell[f, 1]    # no shift needed here as it does not affect area calculation   

                            area_epi = polygon_area(epi_frame)
                            area_endo = polygon_area(endo_frame)

                            # update peak systole frame
                            if area_endo < min_lvc_area:
                                peak_systole_frame = f
                            #update minimum lvc area
                            min_lvc_area = min(min_lvc_area, area_endo)

                            lv_area_frames.append(area_epi - area_endo) # Myocardial Area = |Epi - Endo|
                else:
                    raise ValueError(f"ROIInfo.Contour field not found in slice {slice_name}.")
            except Exception as e:
                print(f"Warning: Error extracting  areas for slice {slice_name}: {e}")
                
        # save lv areas for this slice
        lv_area_slices.append(lv_area_frames)
        slices_peak_systole.append(peak_systole_frame)

    # After processing all slices, finalize the peak systole frame for the volume
    peak_systole_frame_volume = int(np.median(slices_peak_systole))
    print(f"Volume ID: {Vol_id}, Peak Systole Frame (median of slices): {peak_systole_frame_volume}")

    ############################# transfer voxel data to mm data

    # coordinates in mm
    coordinates_mm = np.copy(coordinates_voxel)
    coordinates_mm[:, 0] = coordinates_voxel[:, 0] * voxel_size_x       # x voxel to mm
    coordinates_mm[:, 1] = coordinates_voxel[:, 1] * voxel_size_y       # y voxel to mm
    if data_dimension == 3:
        coordinates_mm[:, 2] = coordinates_voxel[:, 2] * slice_thickness   # z voxel to mm
    

    # lagrangian displacement in mm                           
    lagrangian_displacement_mm = np.copy(lagrangian_displacement_voxel)
    lagrangian_displacement_mm[:, 0, :] = lagrangian_displacement_voxel[:, 0, :] * voxel_size_x     # in-plane voxel x
    lagrangian_displacement_mm[:, 1, :] = lagrangian_displacement_voxel[:, 1, :] * voxel_size_y     # in-plane voxel y
    lagrangian_displacement_mm[:, 2, :] = lagrangian_displacement_voxel[:, 2, :] * voxel_size_z     # in-plane voxel z



    # Epi and Endo contours are in voxel space, convert them to mm space
    endo_contours_mm = []
    epi_contours_mm = []
    for i in range(len(endo_contours)):
        endo = endo_contours[i]
        epi = epi_contours[i]

        endo_mm = np.copy(endo)
        epi_mm = np.copy(epi)

        endo_mm[:, 0] = endo[:, 0] * voxel_size_x
        endo_mm[:, 1] = endo[:, 1] * voxel_size_y

        epi_mm[:, 0] = epi[:, 0] * voxel_size_x
        epi_mm[:, 1] = epi[:, 1] * voxel_size_y

        endo_contours_mm.append(endo_mm)
        epi_contours_mm.append(epi_mm)

    voxel_sizes = (float(voxel_size_x), float(voxel_size_y), float(voxel_size_z))


    return lagrangian_displacement_mm, coordinates_mm, endo_contours_mm, epi_contours_mm, voxel_sizes, lv_area_slices, peak_systole_frame_volume, first_slice_points

        
        


