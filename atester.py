import numpy as np
from scipy.spatial import ConvexHull

def welzl_sphere(points, boundary=[]):
    """
    Computes the smallest enclosing sphere for a given set of points using Welzl's algorithm.
    """
    if len(boundary) == 4 or not points:
        return minimal_sphere(boundary)
    
    p = points[-1]
    sphere = welzl_sphere(points[:-1], boundary)
    
    if np.linalg.norm(p - sphere[0]) > sphere[1]:
        sphere = welzl_sphere(points[:-1], boundary + [p])
    
    return sphere

def minimal_sphere(points):
    """
    Computes the minimal bounding sphere for up to 4 points.
    """
    if not points:
        return np.zeros(3), 0
    elif len(points) == 1:
        return points[0], 0
    elif len(points) == 2:
        center = (points[0] + points[1]) / 2
        radius = np.linalg.norm(points[0] - center)
        return center, radius
    elif len(points) == 3:
        # Compute circumcircle of triangle
        A, B, C = points
        return circumsphere(A, B, C)
    elif len(points) == 4:
        # Compute circumsphere of tetrahedron
        return circumsphere(*points)

def circumsphere(p1, p2, p3, p4):
    """
    Computes the center and radius of the unique sphere passing through four points in 3D.
    """
    A = np.array([
        [p1[0], p1[1], p1[2], 1],
        [p2[0], p2[1], p2[2], 1],
        [p3[0], p3[1], p3[2], 1],
        [p4[0], p4[1], p4[2], 1]
    ])
    
    def det4x4(M):
        return np.linalg.det(M)
    
    D = det4x4(A)
    if abs(D) < 1e-10:
        raise ValueError("The points are coplanar or nearly so; no unique circumsphere exists.")
    
    A_x = A.copy()
    A_x[:, 0] = np.sum(A[:, :3]**2, axis=1)
    A_y = A.copy()
    A_y[:, 1] = np.sum(A[:, :3]**2, axis=1)
    A_z = A.copy()
    A_z[:, 2] = np.sum(A[:, :3]**2, axis=1)
    A_w = A.copy()
    A_w[:, 3] = np.sum(A[:, :3]**2, axis=1)
    
    x_c = det4x4(A_x) / (2 * D)
    y_c = det4x4(A_y) / (2 * D)
    z_c = det4x4(A_z) / (2 * D)
    
    center = np.array([x_c, y_c, z_c])
    radius = np.linalg.norm(center - p1)
    
    return center, radius

# Example usage:
p1 = np.array([1, 0, 0])
p2 = np.array([0, 1, 0])
p3 = np.array([0, 0, 1])
p4 = np.array([1, 1, 1])
p5 = np.array([-1, -1, -1])

points = np.array([p1, p2, p3, p4, p5])
center, radius = welzl_sphere(list(points))
print("Smallest enclosing sphere center:", center)
print("Smallest enclosing sphere radius:", radius)


import numpy as np
from scipy.spatial import Delaunay

def circumsphere_simplex(points):
    """
    Computes the circumsphere (center, radius) of a given simplex (triangle/tetrahedron) in 3D.
    """
    A = np.hstack((points, np.ones((len(points), 1))))
    
    def det(M):
        return np.linalg.det(M)
    
    D = det(A)
    if abs(D) < 1e-10:
        raise ValueError("Degenerate simplex: no unique circumsphere exists.")
    
    A_x = A.copy()
    A_x[:, 0] = np.sum(points**2, axis=1)
    A_y = A.copy()
    A_y[:, 1] = np.sum(points**2, axis=1)
    A_z = A.copy()
    A_z[:, 2] = np.sum(points**2, axis=1)
    A_w = A.copy()
    A_w[:, 3] = np.sum(points**2, axis=1)
    
    x_c = det(A_x) / (2 * D)
    y_c = det(A_y) / (2 * D)
    z_c = det(A_z) / (2 * D)
    
    center = np.array([x_c, y_c, z_c])
    radius = np.linalg.norm(center - points[0])
    
    return center, radius

def alpha_complex(points, alpha):
    """
    Computes the alpha-complex of a given set of points.
    """
    delaunay = Delaunay(points)
    alpha_simplices = []
    
    for simplex in delaunay.simplices:
        simplex_points = points[simplex]
        try:
            center, radius = circumsphere_simplex(simplex_points)
            if radius**2 <= alpha:
                alpha_simplices.append(simplex)
        except ValueError:
            continue
    
    return np.array(alpha_simplices)

# Example usage:
p1 = np.array([1, 0, 0])
p2 = np.array([0, 1, 0])
p3 = np.array([0, 0, 1])
p4 = np.array([1, 1, 1])
p5 = np.array([-1, -1, -1])

points = np.array([p1, p2, p3, p4, p5])
alpha = 1.5  # Example alpha value
simplices = alpha_complex(points, alpha)
print("Alpha complex simplices:", simplices)
