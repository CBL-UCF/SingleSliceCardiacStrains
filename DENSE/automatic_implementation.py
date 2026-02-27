

####################################################################################################################
##########################################        Import Libraries        ########################################## 
####################################################################################################################

from pipeline.mat_loader import data_loader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path
from pipeline.rbf_interpolator import myRBFInterpolator
from pipeline.def_grad_tensor import compute_gradient
from pipeline.transmural_wall_finder import generate_query_points, generate_exact_endo_epi_points



####################################################################################################################
###########################################        Load the Data        ############################################
####################################################################################################################

# Dictionary for subject IDs and the slice they have
# Format: subject_slices = {subject_id: [slice1, slice2], ...}
# Example:

subject_slices = {  
                    4: [5, 6],    # Subject 4, Slices 5 and 6
                    # 9: [7, 8],    # Subject 9, Slices 7 and 8
                    # 11: [7, 8],   # Subject 11, Slices 7 and 8
                    # 22: [4, 5],   # Subject 22, Slices 4 and 5     
                    }

# Slice thickness in mm
slice_thickness_mm = 8.0  # Slice thickness in mm
area_mask_based = False  # Set to True to compute areas from NIFTI masks, False to use areas from epi and endo contours

#####################################################################################################################
##########################################       Main Pipeline Loop        ##########################################
#####################################################################################################################



for vol_id, all_slices in subject_slices.items():

    print(f"\n{'%'*40}")
    print(f"{'%'*40} processing Volume ID: {vol_id} ")
    print(f"{'%'*40}\n")

    senario_one = [all_slices[0]]       # single slice - first slice
    senario_two = [all_slices[1]]       # single slice - second slice
    senario_three = all_slices          # two slices

    for slice_list in [senario_one, senario_two, senario_three]:

        #############################    Print scenario information
        print(f"\n{'='*40}")
        print(f"Running pipeline for Volume ID: {vol_id} slices: {slice_list}")
        print(f"{'='*40}\n")

        #############################    Subject ID string
        subject_id = f"{vol_id:03d}"

        #############################    Load Subject Data
        lagrangian_displacement_mm, coordinates_mm, endo_contours_mm, epi_contours_mm, voxel_sizes, lv_area_slices, peak_systole_frame_volume, first_slice_points = data_loader(Vol_id=vol_id, slice_list=slice_list, area_mask_based=area_mask_based)


        # ###################### Prepare directories for saving results 
        # # Define directories based on subject and slices
        # # e.i. 'results_pipeline/strain/004/5_6/' (for double-slice) or 'results_pipeline/strain/004/5/' (for single slice)

        # if len(slice_list) == 1:
        #     slice_sub_dir = str(slice_list[0])
        # else:
        #     slice_sub_dir = f"{slice_list[0]}_{slice_list[1]}"

        # # Create directories for displacement and strain results
        # lagrangian_disp_dir = f'results_pipeline/displacement/{subject_id}/{slice_sub_dir}/npy'
        # strain_path_subject = f'results_pipeline/strain/{subject_id}/{slice_sub_dir}'

        # for d in [lagrangian_disp_dir, strain_path_subject]:
        #     os.makedirs(d, exist_ok=True)

        # # Save NPY files
        # np.save(os.path.join(lagrangian_disp_dir, 'lagrangian_displacements_mm.npy'), lagrangian_displacement_mm)

        # ############################## Fit RBF
        # kernel = 'cubic'  # 'linear', 'cubic', 'quintic', 'thin_plate'
        # print("\nFitting Global RBF Interpolator...\n")

        # # Check dimensionality based on Z variation
        # is_2d = False
        # print(f'Shape of Coordinates: {coordinates_mm.shape}')

        # # Determine if data is 2D or 3D based on Z-coordinates
        # if coordinates_mm.shape[1] == 2:
        #     is_2d = True
        #     print("Detected single slice (coplanar data). Switching to 2D RBF interpolation.")
        # else:
        #     print("Detected multi-slice data. Using 3D RBF interpolation.")
        
        # # Fit the global RBF interpolator
        # rbf_global = myRBFInterpolator(coordinates_mm, lagrangian_displacement_mm, kernel=kernel) # Default are:  epsilon=1, smoothing=0.01, degree=None) #

        # ############################## Prepare directories and Excel writers for strain results
        # # Excel Writers
        # strain_writer = pd.ExcelWriter(os.path.join(strain_path_subject, 'peak_systole_strain.xlsx'), engine='openpyxl')
        # median_writer = pd.ExcelWriter(os.path.join(strain_path_subject, 'median_strain.xlsx'), engine='openpyxl')
        # all_frames_subdir = os.path.join(strain_path_subject, 'all_frames')
        # os.makedirs(all_frames_subdir, exist_ok=True)

        # ############################## $$$ Create subfolders for transmural strain results $$$ 

        # # Define subfolders for transmural strain results
        # endo_subdir = os.path.join(all_frames_subdir, "endo")
        # mid_subdir = os.path.join(all_frames_subdir, "mid")
        # epi_subdir = os.path.join(all_frames_subdir, "epi")
        # mid_endo_subdir = os.path.join(all_frames_subdir, "mid_endo")
        # mid_epi_subdir = os.path.join(all_frames_subdir, "mid_epi")

        # os.makedirs(endo_subdir, exist_ok=True)
        # os.makedirs(mid_subdir, exist_ok=True)
        # os.makedirs(epi_subdir, exist_ok=True)
        # os.makedirs(mid_endo_subdir, exist_ok=True)
        # os.makedirs(mid_epi_subdir, exist_ok=True)


        # # Define subfolders for exact endo/epi strain results
        # exact_endo_subdir = os.path.join(all_frames_subdir, "exact_endo")
        # exact_epi_subdir = os.path.join(all_frames_subdir, "exact_epi")

        # os.makedirs(exact_endo_subdir, exist_ok=True)
        # os.makedirs(exact_epi_subdir, exist_ok=True)

        # ############################   Global Strain Computation Loop   
        # # Loop over slices in the current scenario so retrieve query points strains
        # for slice_index, slice_num in enumerate(slice_list):
        #     print(f"Querying Slice {slice_num}...")

        #     # Z location of the current slice
        #     z_target = slice_index * slice_thickness_mm 


        #     # Splines
        #     endo_spline = endo_contours_mm[slice_index]
        #     epi_spline = epi_contours_mm[slice_index]

        #     # Define number of segments and angle steps for query points
        #     NUM_SEGMENTS_VAL = 5
        #     NUM_ANGLE_VAL = 5

        #     # Generate radial query points
        #     radial_query_points_all, radial_query_points_endo, radial_query_points_mid, radial_query_points_epi, radial_query_points_mid_endo, radial_query_points_mid_epi = generate_query_points(
        #             endo_spline=endo_spline,
        #             epi_spline=epi_spline,
        #             centroid=np.array([0.0, 0.0]), # as the contours are already origin-centered
        #             num_segments=NUM_SEGMENTS_VAL,
        #             angle_step=NUM_ANGLE_VAL)
            
        #     # Generate EXACT endo/epi points
        #     exact_radial_query_points_endo, exact_radial_query_points_epi = generate_exact_endo_epi_points(
        #             endo_spline=endo_spline,
        #             epi_spline=epi_spline,
        #             centroid=np.array([0.0, 0.0]), # as the contours are already origin-centered
        #             angle_step=NUM_ANGLE_VAL)



        #     ######################## If 3D, add z-coordinate to query points
        #     if not is_2d:
                
        #         # For all radial points
        #         radial_query_points_all = np.column_stack((radial_query_points_all, np.full((len(radial_query_points_all),), z_target)))

        #         # For segmental points
        #         radial_query_points_endo = np.column_stack((radial_query_points_endo, np.full((len(radial_query_points_endo),), z_target)))
        #         radial_query_points_mid = np.column_stack((radial_query_points_mid, np.full((len(radial_query_points_mid),), z_target)))
        #         radial_query_points_epi = np.column_stack((radial_query_points_epi, np.full((len(radial_query_points_epi),), z_target)))
        #         radial_query_points_mid_endo = np.column_stack((radial_query_points_mid_endo, np.full((len(radial_query_points_mid_endo),), z_target)))
        #         radial_query_points_mid_epi = np.column_stack((radial_query_points_mid_epi, np.full((len(radial_query_points_mid_epi),), z_target)))

        #         # Exact points on the endo/epi
        #         exact_radial_query_points_endo = np.column_stack(
        #             (exact_radial_query_points_endo, np.full((len(exact_radial_query_points_endo),), z_target))
        #         )
        #         exact_radial_query_points_epi = np.column_stack(
        #             (exact_radial_query_points_epi, np.full((len(exact_radial_query_points_epi),), z_target))
        #         )
            

            
        #     #############################  List to store median values over time
        #     J_median_global_slice = []
        #     E_rr_median_global_slice = []
        #     E_cc_median_global_slice = []
        #     E_ll_median_global_slice = []
            
        #     # Initialize strain results at peak systole
        #     strain_results_PS = None


        #     # Get the frame number
        #     num_frames = lagrangian_displacement_mm.shape[2]

        #     ###########################   Time Step Loop
        #     for time_step in range(num_frames):
                
        #         # Interpolate displacement
        #         rbf_t = rbf_global.access_rbf(time_step)
                
        #         # Compte Ell for 2D case
        #         if is_2d:

        #             first_frame_area = lv_area_slices[slice_index][0]
        #             frame_area = lv_area_slices[slice_index][time_step]

        #             ell_2d_val = 0.5 * ((first_frame_area / frame_area)**2 - 1) # usign  first frame area

        #         # Compute Gradients & Strains
        #         global_strain_results = {"x": radial_query_points_all[:, 0], "y": radial_query_points_all[:, 1], "J": [], "E_rr": [], "E_cc": [], "E_ll": []}

        #         # Initialize dictionaries for exact endo/epi strain results
        #         exact_endo_strain_results = {"x": [], "y": [], "J": [], "E_rr": [], "E_cc": [], "E_ll": []}
        #         exact_epi_strain_results = {"x": [], "y": [], "J": [], "E_rr": [], "E_cc": [], "E_ll": []}
                
        #         # Use centroid for radial vector calculation as well
        #         centroid_3d = np.array([0, 0, 0]) # as the contours are already origin-centered

        #         # Loop over all radial query points
        #         for i in range(len(radial_query_points_all)):

        #             point = radial_query_points_all[i]

        #             # Compute gradient at the point
        #             grad_matrix = compute_gradient(point.reshape(1, -1), [rbf_t[0], rbf_t[1], rbf_t[2]], kernel=kernel)
        #             F = np.eye(3) + grad_matrix
        #             E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        #             J = np.linalg.det(F)

        #             # Radial Vector
        #             radial_vector = centroid_3d - np.array([point[0], point[1], 0])
        #             norm = np.linalg.norm(radial_vector)
        #             if norm > 1e-6: radial_vector /= norm
                    
        #             circumferential_vector = np.array([-radial_vector[1], radial_vector[0], 0])
        #             longitudinal_vector = np.array([0, 0, 1])

        #             E_rr = radial_vector.T @ E @ radial_vector
        #             E_cc = circumferential_vector.T @ E @ circumferential_vector
                    
        #             # SWITCH LOGIC FOR Ell
        #             if is_2d:
        #                 E_ll = ell_2d_val
        #             else:
        #                 E_ll = longitudinal_vector.T @ E @ longitudinal_vector

        #             global_strain_results["J"].append(J)
        #             global_strain_results["E_rr"].append(E_rr)
        #             global_strain_results["E_cc"].append(E_cc)
        #             global_strain_results["E_ll"].append(E_ll)
                
        #         # Loop over exact endo points
        #         for point in exact_radial_query_points_endo:

        #             # Ensure the point is 3D
        #             if not is_2d:
        #                 point = np.array([point[0], point[1], z_target])  # Add z_target for 3D points

        #             # Compute gradient at the point
        #             grad_matrix = compute_gradient(point.reshape(1, -1), [rbf_t[0], rbf_t[1], rbf_t[2]], kernel=kernel)
        #             F = np.eye(3) + grad_matrix
        #             E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        #             J = np.linalg.det(F)

        #             # Radial and circumferential vectors
        #             radial_vector = centroid_3d - np.array([point[0], point[1], 0])
        #             norm = np.linalg.norm(radial_vector)
        #             if norm > 1e-6:
        #                 radial_vector /= norm

        #             circumferential_vector = np.array([-radial_vector[1], radial_vector[0], 0])
        #             longitudinal_vector = np.array([0, 0, 1])

        #             # Compute strains
        #             E_rr = radial_vector.T @ E @ radial_vector
        #             E_cc = circumferential_vector.T @ E @ circumferential_vector
        #             E_ll = longitudinal_vector.T @ E @ longitudinal_vector if not is_2d else ell_2d_val

        #             # Append results
        #             exact_endo_strain_results["x"].append(point[0])
        #             exact_endo_strain_results["y"].append(point[1])
        #             exact_endo_strain_results["J"].append(J)
        #             exact_endo_strain_results["E_rr"].append(E_rr)
        #             exact_endo_strain_results["E_cc"].append(E_cc)
        #             exact_endo_strain_results["E_ll"].append(E_ll)

        #         # Loop over exact epi points
        #         for point in exact_radial_query_points_epi:

        #             # Ensure the point is 3D
        #             if not is_2d:
        #                 point = np.array([point[0], point[1], z_target])  # Add z_target for 3D points

        #             # Compute gradient at the point
        #             grad_matrix = compute_gradient(point.reshape(1, -1), [rbf_t[0], rbf_t[1], rbf_t[2]], kernel=kernel)
        #             F = np.eye(3) + grad_matrix
        #             E = 0.5 * (np.dot(F.T, F) - np.eye(3))
        #             J = np.linalg.det(F)

        #             # Radial and circumferential vectors
        #             radial_vector = centroid_3d - np.array([point[0], point[1], 0])
        #             norm = np.linalg.norm(radial_vector)
        #             if norm > 1e-6:
        #                 radial_vector /= norm

        #             circumferential_vector = np.array([-radial_vector[1], radial_vector[0], 0])
        #             longitudinal_vector = np.array([0, 0, 1])

        #             # Compute strains
        #             E_rr = radial_vector.T @ E @ radial_vector
        #             E_cc = circumferential_vector.T @ E @ circumferential_vector
        #             E_ll = longitudinal_vector.T @ E @ longitudinal_vector if not is_2d else ell_2d_val

        #             # Append results
        #             exact_epi_strain_results["x"].append(point[0])
        #             exact_epi_strain_results["y"].append(point[1])
        #             exact_epi_strain_results["J"].append(J)
        #             exact_epi_strain_results["E_rr"].append(E_rr)
        #             exact_epi_strain_results["E_cc"].append(E_cc)
        #             exact_epi_strain_results["E_ll"].append(E_ll)
                
        #         # Save exact endo strain results to Excel
        #         df_endo = pd.DataFrame(exact_endo_strain_results)
        #         endo_filename = os.path.join(exact_endo_subdir, f"frame_{time_step+1:02d}.xlsx")

        #         if os.path.exists(endo_filename):
        #             with pd.ExcelWriter(endo_filename, engine='openpyxl', mode='a') as writer:
        #                 # Check if the sheet already exists
        #                 if f"Slice_{slice_num}" not in writer.book.sheetnames:
        #                     df_endo.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)
        #         else:
        #             # Create the file and write the data
        #             with pd.ExcelWriter(endo_filename, engine='openpyxl', mode='w') as writer:
        #                 df_endo.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)

        #         # Save exact epi strain results to Excel
        #         df_epi = pd.DataFrame(exact_epi_strain_results)
        #         epi_filename = os.path.join(exact_epi_subdir, f"frame_{time_step+1:02d}.xlsx")

        #         if os.path.exists(epi_filename):
        #             with pd.ExcelWriter(epi_filename, engine='openpyxl', mode='a') as writer:
        #                 # Check if the sheet already exists
        #                 if f"Slice_{slice_num}" not in writer.book.sheetnames:
        #                     df_epi.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)
        #         else:
        #             # Create the file and write the data
        #             with pd.ExcelWriter(epi_filename, engine='openpyxl', mode='w') as writer:
        #                 df_epi.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)
                
              
                    
                        
        #         ########################################   Saving Global Strain Results

        #         # Convert to arrays
        #         for k in ["J", "E_rr", "E_cc", "E_ll"]:
        #             global_strain_results[k] = np.array(global_strain_results[k])

        #         # Save Frame Excel

        #         df_frame = pd.DataFrame(global_strain_results)
        #         frame_filename = os.path.join(all_frames_subdir, f"frame_{time_step+1:02d}.xlsx")
                
        #         if os.path.exists(frame_filename):
        #             with pd.ExcelWriter(frame_filename, engine='openpyxl', mode='a') as writer:
        #                 if f"Slice_{slice_num}" not in writer.book.sheetnames:
        #                     df_frame.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)
        #         else:
        #             with pd.ExcelWriter(frame_filename, engine='openpyxl', mode='w') as writer:
        #                 df_frame.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)


        #         #######################################   
                
        #         total_pts = len(global_strain_results["x"])
        #         is_endo = np.zeros(total_pts, dtype=bool)
        #         is_mid = np.zeros(total_pts, dtype=bool)
        #         is_epi = np.zeros(total_pts, dtype=bool)
        #         is_mid_endo = np.zeros(total_pts, dtype=bool)
        #         is_mid_epi = np.zeros(total_pts, dtype=bool)
                
        #         # Endo 
        #         is_endo[0::NUM_SEGMENTS_VAL] = True
        #         # Mid 
        #         mid_idx = NUM_SEGMENTS_VAL // 2
        #         is_mid[mid_idx::NUM_SEGMENTS_VAL] = True
        #         # Epi 
        #         is_epi[(NUM_SEGMENTS_VAL-1)::NUM_SEGMENTS_VAL] = True
        #         # Mid-endo 
        #         is_mid_endo[1::NUM_SEGMENTS_VAL] = True
        #         # Mid-epi 
        #         is_mid_epi[3::NUM_SEGMENTS_VAL] = True
                
                
        #         # Assign regions based on boolean masks
        #         endo_mask = is_endo
        #         mid_mask = is_mid
        #         epi_mask = is_epi
        #         mid_endo_mask = is_mid_endo
        #         mid_epi_mask = is_mid_epi

        #         for seg_name, seg_mask, seg_subdir in zip(
        #             ["endo", "mid", "epi", "mid_endo", "mid_epi"],
        #             [endo_mask, mid_mask, epi_mask, mid_endo_mask, mid_epi_mask],
        #             [endo_subdir, mid_subdir, epi_subdir, mid_endo_subdir, mid_epi_subdir]
        #         ):
        #             strain_data_segment = {
        #                 "x": global_strain_results["x"][seg_mask],
        #                 "y": global_strain_results["y"][seg_mask],
        #                 "J": global_strain_results["J"][seg_mask],
        #                 "E_rr": global_strain_results["E_rr"][seg_mask],
        #                 "E_cc": global_strain_results["E_cc"][seg_mask],
        #                 "E_ll": global_strain_results["E_ll"][seg_mask]
        #             }
        #             strain_df_segment = pd.DataFrame(strain_data_segment)
        #             frame_filename_segment = os.path.join(seg_subdir, f"frame_{time_step+1:02d}.xlsx")
                    
        #             if os.path.exists(frame_filename_segment):
        #                 with pd.ExcelWriter(frame_filename_segment, engine='openpyxl', mode='a') as writer:
        #                     if f"Slice_{slice_num}" not in writer.book.sheetnames:
        #                         strain_df_segment.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)
        #             else:
        #                 with pd.ExcelWriter(frame_filename_segment, engine='openpyxl', mode='w') as writer:
        #                     strain_df_segment.to_excel(writer, sheet_name=f"Slice_{slice_num}", index=False)

        #         # Medians
        #         J_median_global_slice.append(np.median(global_strain_results["J"]))
        #         E_rr_median_global_slice.append(np.median(global_strain_results["E_rr"]))
        #         E_cc_median_global_slice.append(np.median(global_strain_results["E_cc"]))
        #         E_ll_median_global_slice.append(np.median(global_strain_results["E_ll"]))

        #         if time_step == peak_systole_frame_volume:
        #             strain_results_PS = global_strain_results.copy()

        #     # Save Peak Systole Data & Plots
        #     peak_systole_subdir = os.path.join(strain_path_subject, 'peak_systole')
        #     os.makedirs(peak_systole_subdir, exist_ok=True)

        #     if strain_results_PS:
        #         strain_data_ps = {
        #             "x": strain_results_PS["x"],
        #             "y": strain_results_PS["y"],
        #             "J": strain_results_PS["J"],
        #             "E_rr": strain_results_PS["E_rr"],
        #             "E_cc": strain_results_PS["E_cc"],
        #             "E_ll": strain_results_PS["E_ll"]
        #         }
                
        #         df_ps = pd.DataFrame(strain_data_ps)
        #         df_ps.to_excel(strain_writer, sheet_name=f"Slice_{slice_num}", index=False)

        #         strain_types = ["J", "E_rr", "E_cc", "E_ll"]
        #         strain_labels = ["Jacobian (J)", "$E_{rr}$", "$E_{cc}$", "$E_{ll}$"]
                
        #         for strain_type, strain_label in zip(strain_types, strain_labels):
        #             plt.figure(figsize=(8, 6))
        #             plt.scatter(strain_data_ps["x"], strain_data_ps["y"], c=strain_data_ps[strain_type], cmap="viridis", s=100, marker='o')
        #             plt.colorbar(label=strain_label)
        #             plt.xlabel("X Coordinate", fontsize=12)
        #             plt.ylabel("Y Coordinate", fontsize=12)
        #             plt.title(f"{strain_label} - Peak Systole - Slice {slice_num}", fontsize=14, fontweight='bold')
        #             plt.tight_layout()
                    
        #             plot_filename = f"{peak_systole_subdir}/{strain_type.lower()}_PS_slice{slice_num}.png"
        #             plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        #             plt.close()
        #             print(f"Peak Systole strain map saved: {plot_filename}")

        #     # Save Medians
        #     median_subdir = os.path.join(strain_path_subject, 'median')
        #     os.makedirs(median_subdir, exist_ok=True)

        #     df_med = pd.DataFrame({
        #         "Time Step": range(num_frames),
        #         "J Median": J_median_global_slice,
        #         "E_rr Median": E_rr_median_global_slice,
        #         "E_cc Median": E_cc_median_global_slice,
        #         "E_ll Median": E_ll_median_global_slice
        #     })
        #     df_med.to_excel(median_writer, sheet_name=f"Slice_{slice_num}", index=False)

        # strain_writer.close()
        # median_writer.close()
        # print("Strain computation complete.")

        # # Plotting Medians
        # cb_palette = sns.color_palette("colorblind", 10)
        # median_file_path = os.path.join(strain_path_subject, 'median_strain.xlsx')
        # median_excel = pd.ExcelFile(median_file_path)
        # sheet_names = median_excel.sheet_names
        # excel_columns = ["J Median", "E_rr Median", "E_cc Median", "E_ll Median"]
        # plot_labels = ["J Median", "$E_{rr}$ Median", "$E_{cc}$ Median", "$E_{ll}$ Median"]

        # for col, label in zip(excel_columns, plot_labels):
        #     plt.figure(figsize=(12,6))
        #     ax = plt.gca()
        #     ax.set_prop_cycle(color=cb_palette)
        #     for sheet in sheet_names:
        #         df = pd.read_excel(median_file_path, sheet_name=sheet)
        #         plt.plot(df["Time Step"], df[col], label=sheet, linewidth=2)
        #     plt.xlabel("Time Step", fontsize=14)
        #     plt.ylabel(label, fontsize=14)
        #     plt.title(f"{label} - {subject_id}", fontsize=16, fontweight='bold')
        #     plt.legend(title="Slice", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        #     plt.tight_layout()
        #     plt.grid()
        #     # plt.show()
            
        #     plot_filename = f"{median_subdir}/{col.replace(' Median', '_median').lower()}.png"
        #     plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        #     plt.close()



