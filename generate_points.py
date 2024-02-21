import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_3d_points_on_plane(num_points, plane_size=1, plane_height=0):    
    # Generate random 2D points on the plane
    points_2d = np.random.rand(num_points, 2) * plane_size

    # Add the constant height to create a 3D plane
    points_3d = np.hstack((points_2d, np.full((num_points, 1), plane_height)))

    return points_3d

def generate_camera_matrices(distance_from_points, separation):
    # Generate the first camera matrix
    camera1_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -distance_from_points],
        [0, 0, 0, 1]
    ])

    # Generate the rotation matrix for the second camera
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.radians(180)), -np.sin(np.radians(180)), 0],
        [0, np.sin(np.radians(180)), np.cos(np.radians(180)), 0],
        [0, 0, 0, 1]
    ])

    # Generate the translation matrix for the second camera
    translation_matrix = np.array([
        [1, 0, 0, separation],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Calculate the second camera matrix by combining rotation and translation
    camera2_matrix = translation_matrix @ rotation_matrix @ camera1_matrix

    return camera1_matrix, camera2_matrix

def transform_points_to_camera_coordinates(points, camera_matrix):
    # Homogeneous coordinates for the 3D points
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform points to camera coordinates
    points_camera_coordinates = (camera_matrix @ points_homogeneous.T).T[:, :3]

    return points_camera_coordinates

def main():
    # Parameters
    num_points = 50
    distance_from_points = 2  # Distance from cameras to the points
    separation = 1  # Separation between the cameras

    # Generate 3D points
    points = generate_3d_points_on_plane(num_points)

    # Generate camera matrices
    camera1_matrix, camera2_matrix = generate_camera_matrices(distance_from_points, separation)

    # Transform points to camera coordinates
    points_camera1 = transform_points_to_camera_coordinates(points, camera1_matrix)
    points_camera2 = transform_points_to_camera_coordinates(points, camera2_matrix)

    # Plot 3D points in both camera coordinate systems
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_camera1[:, 0], points_camera1[:, 1], points_camera1[:, 2])
    ax1.set_title('Camera 1 Coordinates')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points_camera2[:, 0], points_camera2[:, 1], points_camera2[:, 2])
    ax2.set_title('Camera 2 Coordinates')

    plt.show()

if __name__ == "__main__":
    main()