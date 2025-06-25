# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 08:43:50 2025

@author: Mikhail
"""

import CSPointCloud as CSPC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def trimPC(path, trimpoints = 100, show=False):
    """ This function trims the amount of points of a point cloud stored in the 'path' directory to 'trimpoints'. """
    pc_array = CSPC.pcToArray(path)
    if(pc_array.shape[0] <= trimpoints):
        print('Warning: trimpoints is more than the original length. Point Cloud is not trimmed!')
    trim = pc_array.shape[0] - trimpoints 
    pc_array_trim = pc_array[:-trim, :]
    pc_trim = CSPC.arrayToPC(pc_array_trim)
    if(show):
        pc_trim.showPC()
    return pc_trim

def wavelet_reconstruct(tile_points):
    # Placeholder: no-op or noisy version of input
    # Replace with your actual sparse wavelet recovery method
    return tile_points + np.random.normal(scale=0.001, size=tile_points.shape)

def sliding_wavelet_reconstruction(point_cloud, tile_size=1.0, overlap_fraction=0.5):
    """
    Parameters:
        point_cloud: (N, 3) numpy array of 3D coordinates
        tile_size: float, size of the cube tile along each axis
        overlap_fraction: float in [0, 1), how much overlap between tiles
    Returns:
        Reconstructed and stitched point cloud as numpy array (M, 3)
    """

    # Step 1: Bounding box of the point cloud
    min_coords = point_cloud.min(axis=0)
    max_coords = point_cloud.max(axis=0)

    stride = tile_size * (1 - overlap_fraction)
    if stride <= 0:
        raise ValueError("Overlap fraction too large; stride must be positive.")

    # Step 2: Generate tile origins (grid sliding)
    x_starts = np.arange(min_coords[0], max_coords[0], stride)
    y_starts = np.arange(min_coords[1], max_coords[1], stride)
    z_starts = np.arange(min_coords[2], max_coords[2], stride)

    # Storage for stitched reconstruction
    weight_map = {}  # key: tuple rounded coord, value: (sum, weight_sum)

    for x0 in x_starts:
        for y0 in y_starts:
            for z0 in z_starts:
                origin = np.array([x0, y0, z0])
                bounds_min = origin
                bounds_max = origin + tile_size

                # Step 3: Extract points in this tile
                in_tile_mask = np.all((point_cloud >= bounds_min) & (point_cloud <= bounds_max), axis=1)
                tile_points = point_cloud[in_tile_mask]

                if len(tile_points) < 10:
                    continue  # Skip very sparse tiles

                # Step 4: Reconstruct tile (your wavelet-based method goes here)
                reconstructed_tile = wavelet_reconstruct(tile_points)  # Should return (M, 3) array
                #reconstructed_tile = None

                # Step 5: Gaussian-weighted stitching
                center = origin + tile_size / 2
                sigma = tile_size / 4

                distances = np.linalg.norm(reconstructed_tile - center, axis=1)
                weights = np.exp(- (distances**2) / (2 * sigma**2))

                for pt, w in zip(reconstructed_tile, weights):
                    key = tuple(np.round(pt, decimals=3))  # 3 decimal places → ~mm resolution
                    if key in weight_map:
                        acc_sum, acc_w = weight_map[key]
                        weight_map[key] = (acc_sum + pt * w, acc_w + w)
                    else:
                        weight_map[key] = (pt * w, w)

    # Step 6: Normalize weights to get final stitched point cloud
    final_points = [sum_pt / weight for sum_pt, weight in weight_map.values()]
    return np.vstack(final_points)




if __name__ == "__main__":
    ''' Trim points of a point cloud '''
    '''
    pc1 = CSPC.CSPCdwt()
    pc1.setLasCoords(pcpath='D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\downsampled\\Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las')
    pc1.showPC()
    pc2 = trimPC(path='D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\downsampled\\Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', trimpoints=1000)
    pc2.showPC()
    
    pc1_array = np.stack([pc1.x, pc1.y, pc1.z], axis=-1)
    pc2_array = np.stack([pc2.x, pc2.y, pc2.z], axis=-1)
    
    plotPC(pc1.x, pc1.y, pc1.z, title='Original', e_angle=35, a_angle=-205)
    plotPC(pc2.x, pc2.y, pc2.z, title='Trimmed', e_angle=35, a_angle=-205)
    '''
    
    ''' Stitch two point clouds together '''
    '''
    cloud1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ])
    
    cloud2 = np.array([
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0]
    ])
    
    # Combine them into one
    combined_cloud = np.vstack((cloud1, cloud2))
    
    print("Combined Point Cloud:")
    print(combined_cloud)
    '''
    
    
    ''' Partitioning + Stitching Tiles With Colors '''
    
    #pc1_arr = CSPC.pcToArray('D:\\Documents\\Thesis_CS\\Point_Cloud_Inputs\\helix.las')
    pc1_arr = CSPC.pcToArray('D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\downsampled\\Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las')

    # --- Generate Synthetic Point Cloud ---
    np.random.seed(42)
    point_cloud = pc1_arr

    tile_size = 100 # Space between each point in tile


    # --- Tile Split & Colored Reconstruct ---
    min_coords = point_cloud.min(axis=0)
    max_coords = point_cloud.max(axis=0)

    # Get axis ranges
    #axis_ranges = max_coords - min_coords
    
    # Calculate number of tiles per axis
    #num_tiles = np.ceil(axis_ranges / tile_size).astype(int)    

    # 
    x_starts = np.arange(min_coords[0], max_coords[0], tile_size)
    y_starts = np.arange(min_coords[1], max_coords[1], tile_size)
    z_starts = np.arange(min_coords[2], max_coords[2], tile_size)

    reconstructed_tiles = [] # List of numpy arrays that contain reconstructed coordinates of each tile
    tile_colors = []

    for x0 in x_starts:
        for y0 in y_starts:
            for z0 in z_starts:
                # Starting point of each tile
                origin = np.array([x0, y0, z0])
                bounds_min = origin
                bounds_max = origin + tile_size
                
                # Check if entire point cloud is within the tile bounds (create boolean mask)
                in_tile_mask = np.all((point_cloud >= bounds_min) & (point_cloud < bounds_max), axis=1)
                tile_points = point_cloud[in_tile_mask] # Select only the points inside the current tile by applying the boolean mask.

                if tile_points.shape[0] == 0:
                    continue

                # Mock reconstruction
                reconstructed_tile = tile_points + np.random.normal(scale=10, size=tile_points.shape)
                #reconstructed_tile = tile_points
                reconstructed_tiles.append(reconstructed_tile)

                # Assign a random RGB color to all points in this tile
                random_color = np.random.rand(3)
                tile_colors.append(np.tile(random_color, (reconstructed_tile.shape[0], 1)))

    # Stack all tiles and their corresponding colors
    combined_cloud = np.vstack(reconstructed_tiles)
    combined_colors = np.vstack(tile_colors)

    # --- Visualization ---
    fig = plt.figure(figsize=(14, 6))

    # Original Point Cloud
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                c='gray', s=5)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlim(min(point_cloud[:,0]), max(point_cloud[:,0]))
    ax1.set_ylim(min(point_cloud[:,1]), max(point_cloud[:,1]))
    ax1.set_zlim(min(point_cloud[:,2]), max(point_cloud[:,2]))

    # Colored Tiles
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(combined_cloud[:, 0], combined_cloud[:, 1], combined_cloud[:, 2],
                c=combined_colors, s=5)
    ax2.set_title('Reconstructed Tiles Colored by Region')
    ax2.set_xlim(min(point_cloud[:,0]), max(combined_cloud[:,0]))
    ax2.set_ylim(min(point_cloud[:,1]), max(combined_cloud[:,1]))
    ax2.set_zlim(min(point_cloud[:,2]), max(combined_cloud[:,2]))

    plt.tight_layout()
    plt.show()
    
    
    
    
    ''' Partitioning + Stitching Tiles With Colors and With Bounding Boxes '''
    
    '''
    # --- Generate Synthetic Point Cloud ---
    np.random.seed(42)
    point_cloud = np.random.uniform(0, 10, size=(5000, 3))
    
    tile_size = 5.0
    min_coords = point_cloud.min(axis=0)
    max_coords = point_cloud.max(axis=0)
    
    x_starts = np.arange(min_coords[0], max_coords[0], tile_size)
    y_starts = np.arange(min_coords[1], max_coords[1], tile_size)
    z_starts = np.arange(min_coords[2], max_coords[2], tile_size)
    
    reconstructed_tiles = []
    tile_colors = []
    tile_boxes = []
    
    for x0 in x_starts:
        for y0 in y_starts:
            for z0 in z_starts:
                origin = np.array([x0, y0, z0])
                bounds_min = origin
                bounds_max = origin + tile_size
    
                in_tile_mask = np.all((point_cloud >= bounds_min) & (point_cloud < bounds_max), axis=1)
                tile_points = point_cloud[in_tile_mask]
    
                if tile_points.shape[0] == 0:
                    continue
    
                # Mock reconstruction
                reconstructed_tile = tile_points + np.random.normal(scale=0.01, size=tile_points.shape)
                reconstructed_tiles.append(reconstructed_tile)
    
                # Random color per tile
                color = np.random.rand(3)
                tile_colors.append(np.tile(color, (reconstructed_tile.shape[0], 1)))
    
                # Save box bounds for visualization
                tile_boxes.append((bounds_min, bounds_max))
    
    combined_cloud = np.vstack(reconstructed_tiles)
    combined_colors = np.vstack(tile_colors)
    
    # --- Bounding Box Helper ---
    def get_cube_faces(bounds_min, bounds_max):
        x0, y0, z0 = bounds_min
        x1, y1, z1 = bounds_max
        return [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
            [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],  # left
            [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],  # right
            [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],  # front
            [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],  # back
        ]
    
    # --- Visualization ---
    fig = plt.figure(figsize=(14, 6))
    
    # Original Point Cloud
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                c='gray', s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_zlim(0, 10)
    
    # Colored Tiles + Bounding Boxes
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(combined_cloud[:, 0], combined_cloud[:, 1], combined_cloud[:, 2],
                c=combined_colors, s=1)
    ax2.set_title('Reconstructed Tiles with Bounding Boxes')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_zlim(0, 10)
    
    print(f"Number of tiles: {len(tile_boxes)}")
    print(f"First box (bounds_min, bounds_max): {tile_boxes[0]}")

    
    # Draw bounding boxes
    for bounds_min, bounds_max in tile_boxes:
        cube_faces = get_cube_faces(bounds_min, bounds_max)
        box = Poly3DCollection(cube_faces, facecolors='none', edgecolors='black', linewidths=0.5)
        ax2.add_collection3d(box)
    
    plt.tight_layout()
    plt.show()
    '''

        
    
    