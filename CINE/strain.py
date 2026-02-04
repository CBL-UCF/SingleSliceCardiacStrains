import numpy as np
from skimage.measure import  find_contours
from scipy.interpolate import make_lsq_spline
from scipy.integrate import quad

def barycenter(mask):
    """
    Compute the center of the mask using the barycenter approach

    Args:
        mask (2-D array): The prediction mask representing the LVM

    Returns:
        x_c (float): The x location of the center of the mask
        y_c (float): The y location of the center of the mask
    """
    # Transform the mask in float to be able to get decimal division
    mask = mask.astype(float)
    
    # Sum the mask
    total_mass = mask.sum()
    if total_mass == 0:
        return None, None
    
    # Get correct indices for rows and columns from the array axis (x, y flipped)
    rows, cols = np.indices(mask.shape)
    
    # Compute barycenter
    y_c = (rows * mask).sum() / total_mass  # row = y
    x_c = (cols * mask).sum() / total_mass  # col = x
    
    return x_c, y_c

def green_lagrange(numerator, denominator):
    """
    Apply Green Lagrange approximation formula

    Args:
        numerator (float): numerator of the lambda ratio
        denominator (float): denominator of the lambda ratio

    Returns:
        (float): computed Green Lagrangian Strain 
    """
    # Computing lambda ratio
    l = numerator / denominator

    # Green Lagrange formula
    return 0.5 * (l ** 2 - 1)

def compute_ell(pred_array):
    """
    Compute longitudinal strain based on the predicted mask

    Args:
        pred_array (list): list of all predicted masks for given slice

    Returns:
        ells (list): list of longitudinal strain for every frame 
    """

    # List that will hold results and reference frame area
    ells = []
    area_ref = 0

    # Looping through each predicted mask
    for t in range(len(pred_array)):

        # Computing area in reference frame and current frame
        if t == 0:
            area_ref = np.sum(pred_array[t])
        area_t = np.sum(pred_array[t])

        # Get Ell at current frame and add to the res list
        Ell = green_lagrange(area_ref, area_t)
        ells.append(Ell)
    
    return ells

def get_polar_coord(contour, xc, yc):
    """
    Compute polar coordinates from a given contour

    Args:
        contour (2-D array, floats): contour of the predicted mask
        xc (float): x coordinate of barycenter of mask
        yc (float): y coordinate of barycenter of mask

    Returns:
        r (array, float): list of radii based on the x,y locations of the contour
        theta (array, float): angles for given raddi in radians
    """
    x = contour[:,1]
    y = contour[:,0]
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    theta = np.arctan2(y-yc, x-xc)
    return r, theta

def build_spline(theta, r, n_knots=17, k=3):
    """
    Function that builds the cubic spline from given polar coordinate

    Args:
        theta (float): angles from contour in radians
        r (float): radii from contour
        n_knots (int): number of knots, should be one more than what expected due to 
            repeating a point at first and last points (e.g 17 = 16 pts in spline (0,360) are the same points)
        k (int): degree of the spline, since we need a cubic spline it is always 3

    Returns:
        (function): function that returns r based on theta for the built spline
    """
    # Sorting theta and getting r of sorted values
    idx = np.argsort(theta)
    theta, r = theta[idx], r[idx]

    # The contour does not give exactly -pi as start and end points. Therefore, add those to the list for full rotation
    theta = np.concatenate(([-np.pi], theta, [np.pi]))

    # Compute the radius for -pi
    r_pi = np.mean([r[0], r[-1]])
    r = np.concatenate(([r_pi], r, [r_pi]))
    
    # Getting the angle between each knot
    knot_spacing = (2 * np.pi) / (n_knots - 1)
    pad_angle = knot_spacing * (k + 1)/2
   
    # Getting indices of angles one knot after the first knots and two before the last knot
    left_pad_mask = theta < (theta[0] + pad_angle/2)
    right_pad_mask = theta > (theta[-1] - pad_angle)
    
    # Adding those values for theta and r for representing a loop for the library
    theta_padded = np.concatenate([
        theta[right_pad_mask] - 2*np.pi, 
        theta, 
        theta[left_pad_mask] + 2*np.pi
    ])
    r_padded = np.concatenate([
        r[right_pad_mask], 
        r, 
        r[left_pad_mask]
    ])

    # Setting start and end points following the SciPy rules for boundaries
    t_min, t_max = theta_padded[0], theta_padded[-1]
    inner_knots = np.linspace(t_min, t_max, n_knots)[1:-1]
    knots = np.concatenate(([t_min] * (k + 1), inner_knots, [t_max] * (k + 1)))

    # Create the spline functon
    return make_lsq_spline(theta_padded, r_padded, knots, k=k)

def compute_circumference(spline):
    """
    Compute length of the circumference of the spline

    Args:
        spline (2-D array, float): polar coordinates points of the result spline
    Returns:
        circumference (float): length of circumference of the spline
    """
    # Define the integrand function: sqrt(r^2 + (dr/dtheta)^2)
    def integrand(theta):
        r = spline(theta)
        # derivative() returns a new spline representing the derivative
        dr_dtheta = spline.derivative()(theta)
        return np.sqrt(r**2 + dr_dtheta**2)

    # Numerically integrate from -pi to pi
    circumference, error = quad(integrand, -np.pi, np.pi)
    return circumference


def compute_strain_least_sqr(pred_array):
    """
    Compute radial and circumferential strains

    Args:
        pred_array (list): list of arrays holding the predicted mask for every frame
    Returns:
        errs (list): radial strain for every frame
        eccs_epi (list): circumferential-epicardium strain for every frame 
        eccs_endo (list): circumferential-endocardium strain for every frame
    """

    # Angle list for that will be used to compute the spline (every 1 deg)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=True )

    # Err reference thickness and errs result list
    t_ref = None
    errs = []

    # Ecc reference circumferences and result lists (endo & epi)
    eccs_endo = []
    eccs_epi = []
    c_ref_epi = None
    c_ref_endo = None

    # Iterate over prediction masks
    for i, pred in enumerate(pred_array):
        # Get the contour for the mask
        contours = find_contours(pred, level=0.5)
        
        # Compute contour circumference to be able to find endo and epi
        circumferences = [np.sum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))) for contour in contours]

        endo_c = min(circumferences)
        epi_c = max(circumferences)

        idx_max = circumferences.index(epi_c)
        idx_min = circumferences.index(endo_c)

        C_epi_raw = contours[idx_max]
        C_endo_raw = contours[idx_min]

        # Find the barycenter
        xc, yc = barycenter(pred)

        # Get contour radii and angles
        r_endo_raw, theta_endo_raw = get_polar_coord(C_endo_raw, xc, yc)
        r_epi_raw, theta_epi_raw = get_polar_coord(C_epi_raw, xc, yc)

        theta_endo_sorted_idx = np.argsort(theta_endo_raw)
        theta_epi_sorted_idx = np.argsort(theta_epi_raw)

        r_endo_raw = r_endo_raw[theta_endo_sorted_idx]
        theta_endo_raw = theta_endo_raw[theta_endo_sorted_idx]

        r_epi_raw = r_epi_raw[theta_epi_sorted_idx]
        theta_epi_raw = theta_epi_raw[theta_epi_sorted_idx]

        # Build the splie for both endo and epi
        endo_spline = build_spline(theta_endo_raw, r_endo_raw)
        epi_spline = build_spline(theta_epi_raw, r_epi_raw)

        # Get the radii for all angles (every 1 degree)
        r_endo = endo_spline(angles)
        r_epi = epi_spline(angles)

        r_endo[-1] = r_endo[0]
        r_epi[-1] = r_epi[0]

        # Get the circumference for end and epi
        endo_circ = compute_circumference(endo_spline)
        epi_circ = compute_circumference(epi_spline)

        # Thickness
        t_current = r_epi - r_endo

        # Reference thickness (BOS)
        if i == 0:
            t_ref = t_current.copy()
            c_ref_endo = endo_circ
            c_ref_epi = epi_circ

        # Green–Lagrange radial strain
        Err = green_lagrange(t_current, t_ref)
        eccs_endo.append(green_lagrange(endo_circ, c_ref_endo))
        eccs_epi.append(green_lagrange(epi_circ, c_ref_epi))


        # Global Err (robust)
        errs.append(np.median(Err))

    return errs, eccs_epi, eccs_endo
        

def compute_strain_cine(pred_array):
    """
    Pipeline function

    Args:
        pred_array (list): list of arrays holding the predicted mask for every frame
    Returns:
        errs (list): radial strain for every frame
        ells (list): longitudinal strain for every frame
        eccs_epi (list): circumferential-epicardium strain for every frame 
        eccs_endo (list): circumferential-endocardium strain for every frame
    """
    errs, eccs_epi, eccs_endo = compute_strain_least_sqr(pred_array)
    ells = compute_ell(pred_array)

    return errs, ells, eccs_epi, eccs_endo 

