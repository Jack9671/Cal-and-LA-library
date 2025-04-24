import numpy as np
import math
import GraphPloting as gp
#set the precision
class MatrixLibrary:
    def __init__(self):
        """Initialize the MatrixLibrary with common matrices or leave empty if only functions are used."""
        pass

    def Rotation_matrix(self, angle, axis="not chosen", art_axis=np.array([0, 0, 0])) -> np.array:
        """Create a 2D or 3D rotation matrix."""
        angle_rad = math.radians(angle)
        
        if axis == 'z':  # Rotate around the z-axis (3D rotation or 2D rotation)
            return np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        elif axis == 'y':  # Rotate around the y-axis (3D rotation)
            return np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'x':  # Rotate around the x-axis (3D rotation)
            return np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        else: # Rotate around an arbitrary axis (3D rotation)
            art_axis = art_axis.flatten()
            u = art_axis / np.linalg.norm(art_axis)  # Normalize the axis
            ux, uy, uz = u
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)
            return np.array([
                [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
                [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
                [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
            ])

    
    def Translation_matrix(self, dx, dy, dz):
        """Create a 3D translation matrix."""
        return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
        ])