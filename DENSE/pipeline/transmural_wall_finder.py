import numpy as np


def find_intersection(centroid, direction, spline):
    """
    Find the intersection of a radial line with a spline.
    """
    # Ensure inputs are float64 to avoid object dtype errors in linalg.solve
    centroid = np.array(centroid, dtype=np.float64)
    direction = np.array(direction, dtype=np.float64)
    spline = np.array(spline, dtype=np.float64)

    for i in range(len(spline) - 1):
        # Define the segment of the spline
        p1, p2 = spline[i], spline[i + 1]

        # Solve for intersection between the radial line and the spline segment
        A = np.array([direction, p1 - p2]).T
        b = p1 - centroid
        
        # Double check A is float
        A = A.astype(np.float64)
        b = b.astype(np.float64)

        try:
            t, u = np.linalg.solve(A, b)
            if 0 <= u <= 1 and t >= 0:
                # Intersection point
                return centroid + t * direction
        except np.linalg.LinAlgError:
            # Lines are parallel, no intersection
            continue
    return None

def generate_query_points(endo_spline, epi_spline, centroid, num_segments=5, angle_step=10):
    """
    Generate query points along radial lines from the centroid of the myocardium.
    Ensures points are generated from Inner (Endo) to Outer (Epi).
    """
    query_points = []
    endo_points = []
    mid_points = []
    epi_points = []
    mid_endo_points = []
    mid_epi_points = []
    

    for angle in range(0, 360, angle_step):
        theta = np.radians(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        endo_intersection = find_intersection(centroid, direction, endo_spline)
        epi_intersection = find_intersection(centroid, direction, epi_spline)

        if endo_intersection is not None and epi_intersection is not None:
            #  Determine which point is Inner vs Outer based on distance 
            dist_endo = np.linalg.norm(endo_intersection - centroid)
            dist_epi = np.linalg.norm(epi_intersection - centroid)

            if dist_endo < dist_epi:
                p_inner = endo_intersection
                p_outer = epi_intersection
            else:
                p_inner = epi_intersection
                p_outer = endo_intersection
            
            # Generate points from Inner -> Outer
            segment_points = np.linspace(p_inner, p_outer, num_segments + 1)
            # --------------------------------------------------------------------

            mid_idx = num_segments // 2  # Only valid for odd num_segments

            for i in range(num_segments):
                midpoint = (segment_points[i] + segment_points[i + 1]) / 2
                query_points.append(midpoint)
                if num_segments % 2 == 1:
                    if i == 0:
                        endo_points.append(midpoint) 
                    elif num_segments == 5 and i == 1:
                        mid_endo_points.append(midpoint)
                    elif i == mid_idx:
                        mid_points.append(midpoint)
                    elif num_segments == 5 and i == 3:
                        mid_epi_points.append(midpoint)
                    elif i == num_segments - 1:
                        epi_points.append(midpoint) 

    return np.array(query_points), np.array(endo_points), np.array(mid_points), np.array(epi_points), np.array(mid_endo_points), np.array(mid_epi_points)


def generate_exact_endo_epi_points(endo_spline, epi_spline, centroid, angle_step=10):
    """
    Generate exact intersection points of radial lines with the endocardial and epicardial splines.
    
    Args:
        endo_spline (np.ndarray): Coordinates of the endocardial spline.
        epi_spline (np.ndarray): Coordinates of the epicardial spline.
        centroid (np.ndarray): The centroid of the myocardium.
        angle_step (int): Step size for angles in degrees (default: 10).

    Returns:
        np.ndarray: Exact intersection points with the endocardial spline.
        np.ndarray: Exact intersection points with the epicardial spline.
    """
    endo_points = []
    epi_points = []

    for angle in range(0, 360, angle_step):
        theta = np.radians(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        
        # Find exact intersection points with the endo and epi splines
        endo_intersection = find_intersection(centroid, direction, endo_spline)
        epi_intersection = find_intersection(centroid, direction, epi_spline)

        if endo_intersection is not None:
            endo_points.append(endo_intersection) 
        if epi_intersection is not None:
            epi_points.append(epi_intersection)

    return np.array(endo_points), np.array(epi_points)
