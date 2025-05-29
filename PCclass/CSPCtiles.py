# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 19:35:32 2025

@author: Mikhail
"""


import CSPointCloud as CSPC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def generate_tile_starts(min_coord, max_coord, tile_size, min_tile_size):
    axis_range = max_coord - min_coord
    num_full_tiles = int(axis_range // tile_size)
    leftover = axis_range % tile_size

    if leftover < min_tile_size and num_full_tiles > 1:
        # Not enough leftover to form a good tile: squeeze into fewer tiles
        num_full_tiles -= 1

    # Now generate linearly spaced starts
    starts = np.linspace(min_coord, min_coord + tile_size * (num_full_tiles - 1), num=num_full_tiles)
    return starts




def reconstruct_tile(tile_points, cs_ratio=0.15, sparsity_val=90, measurement_type='gaussian', wvlt = 'haar', parallel=True):

    '''
        1. pc = CSPC.CSPCdwt()
        2. pc.setManualCoords(tile_points[:,0], tile_points[:,1], tile_points[:,2])
        3. thresholding, measurement, reconstruction
        4. reconstructed_tile = np.stack((pc.x_r, pc.y_r, pc.z_r), axis=1)
        FOR NOW: Ignore individual results (errors, solvertimes) of each tile. Only obtain them as a whole at the end.
    '''
    
    pc = CSPC.CSPCdwt()
    pc.setManualCoords(tile_points[:,0], tile_points[:,1], tile_points[:,2])
    
    fcx, cx, fcy, cy, fcz, cz = pc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    fcx_th, thx = CSPC.applyThresholding(fcx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep (nonzer0)
    fcy_th, thy = CSPC.applyThresholding(fcy, percentile=sparsity_percentile, type_th ='hard')
    fcz_th, thz = CSPC.applyThresholding(fcz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
        #s, sn = CSPC.calculateSparsity(fcx, fcy, fcz)
    s, sn = CSPC.calculateSparsity(fcx_th, fcy_th, fcz_th)
    # Create measurement matrix
    n_coeffs = len(fcx_th)
        #n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx_th, y_flat_coeffs=fcy_th, z_flat_coeffs=fcz_th)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    if(parallel):
        solvertime = pc.reconstructCVXPY_ray(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
    else:
        solvertime = pc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)    
    
    reconstructed_tile = np.stack((pc.x_r, pc.y_r, pc.z_r), axis=1)
    print('Solver time:-----------------------')
    print(solvertime)
    return reconstructed_tile, solvertime


def run_tile_reconstruction(in_path='D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\downsampled\\Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las'):
    
    pc1_arr = CSPC.pcToArray(in_path)

    # --- Generate Synthetic Point Cloud ---
    np.random.seed(42)
    point_cloud = pc1_arr

    # TODO: WARNING: IF TILE_SIZE IS TOO SMALL, CVXPY RECONSTRUCTION DOES NOT WORK SINCE
    # SOME TILES MAY RESULT IN ONLY BEING ONE OR TWO POINTS
    
    tile_size = 100 # (Each tile is spaced 'tile_size" units apart)


    # --- Tile Split & Colored Reconstruct ---
    min_coords = point_cloud.min(axis=0)
    max_coords = point_cloud.max(axis=0)

    # Get axis ranges
    axis_ranges = max_coords - min_coords
    
    # Calculate number of tiles per axis
    num_tiles = np.ceil(axis_ranges / tile_size).astype(int)    

    # Generate linearly spaced starting positions (excluding last edge)
    '''
    x_starts = np.linspace(min_coords[0], min_coords[0] + tile_size * (num_tiles[0] - 1), num=num_tiles[0])
    y_starts = np.linspace(min_coords[1], min_coords[1] + tile_size * (num_tiles[1] - 1), num=num_tiles[1])
    z_starts = np.linspace(min_coords[2], min_coords[2] + tile_size * (num_tiles[2] - 1), num=num_tiles[2])
    '''
    
    x_starts = generate_tile_starts(min_coords[0], max_coords[0], tile_size, min_tile_size=5)
    y_starts = generate_tile_starts(min_coords[1], max_coords[1], tile_size, min_tile_size=5)
    z_starts = generate_tile_starts(min_coords[2], max_coords[2], tile_size, min_tile_size=5)


    ''' '''
    #x_starts = np.arange(min_coords[0], max_coords[0], tile_size)
    #y_starts = np.arange(min_coords[1], max_coords[1], tile_size)
    #z_starts = np.arange(min_coords[2], max_coords[2], tile_size)

    reconstructed_tiles = [] # List of numpy arrays that contain reconstructed coordinates of each tile
    tile_colors = []

    for x0 in x_starts:
        for y0 in y_starts:
            for z0 in z_starts:
                # Starting point of each tile
                origin = np.array([x0, y0, z0])
                bounds_min = origin
                bounds_max = origin + tile_size
                
                # Makes boolean matrix that shows indices where the point cloud's x,y, and z coords
                # are within the tile bounds.
                # Then, along each coordinate point, use np.all to see which coordinate has all 3 "Trues", meaning
                # that coordinate will be the one selected in tile_points
                # Check if entire point cloud is within the tile bounds (create boolean mask)
                in_tile_mask = np.all((point_cloud >= bounds_min) & (point_cloud < bounds_max), axis=1)
                tile_points = point_cloud[in_tile_mask] # Select only the points inside the current tile by applying the boolean mask.

                if tile_points.shape[0] == 0:
                    continue

                # Mock reconstruction
                #reconstructed_tile = tile_points + np.random.normal(scale=0.01, size=tile_points.shape)
                #reconstructed_tile = tile_points
                
                # Reconstruction
                reconstructed_tile = reconstruct_tile(tile_points)
                print(reconstructed_tile)
                
                
                
                '''
                    1. Create Point CLoud Object
                    2. Set LAS coords from tile_points in the PC object
                    3. Do the whole reconstruction procedure (input reconstruction parameters)
                    4. Without showing/plotting the PC tile or outputting reconstruction errors,
                       output only the reconstructed PC coords as (m x 3) ndarray
                    FOR NOW: Ignore individual results (errors, solvertimes) of each tile. Only obtain them as a whole at the end.
                '''
                
                '''
                    1. pc = CSPC.CSPCdwt()
                    2. pc.setManualCoords(tile_points[:,0], tile_points[:,1], tile_points[:,2])
                    3. thresholding, measurement, reconstruction
                    4. reconstructed_tile = np.stack((pc.x_r, pc.y_r, pc.z_r), axis=1)
                    FOR NOW: Ignore individual results (errors, solvertimes) of each tile. Only obtain them as a whole at the end.
                '''
                
                reconstructed_tiles.append(reconstructed_tile)

                # Assign a random color to all points in each tile
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
    ax2.set_xlim(min(combined_cloud[:,0]), max(combined_cloud[:,0]))
    ax2.set_ylim(min(combined_cloud[:,1]), max(combined_cloud[:,1]))
    ax2.set_zlim(min(combined_cloud[:,2]), max(combined_cloud[:,2]))

    plt.tight_layout()
    plt.show()
    
    '''
    ADD RECONSTRUCTION ERRORS/INFO HERE
    '''
    
if __name__ == "__main__":
    '''
    run_tile_reconstruction()
    '''
    
    pc1_arr = CSPC.pcToArray(pcpath='D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\downsampled\\Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las')

    # --- Generate Synthetic Point Cloud ---
    np.random.seed(42)
    point_cloud = pc1_arr

    # TODO: WARNING: IF TILE_SIZE IS TOO SMALL, CVXPY RECONSTRUCTION DOES NOT WORK SINCE
    # SOME TILES MAY RESULT IN ONLY BEING ONE OR TWO POINTS
    
    tile_size = 100 # (Each tile is spaced 'tile_size" units apart)


    # --- Tile Split & Colored Reconstruct ---
    min_coords = point_cloud.min(axis=0)
    max_coords = point_cloud.max(axis=0)

    # Get axis ranges
    axis_ranges = max_coords - min_coords
    
    # Calculate number of tiles per axis
    num_tiles = np.ceil(axis_ranges / tile_size).astype(int)    

    # Generate linearly spaced starting positions (excluding last edge)
    '''
    x_starts = np.linspace(min_coords[0], min_coords[0] + tile_size * (num_tiles[0] - 1), num=num_tiles[0])
    y_starts = np.linspace(min_coords[1], min_coords[1] + tile_size * (num_tiles[1] - 1), num=num_tiles[1])
    z_starts = np.linspace(min_coords[2], min_coords[2] + tile_size * (num_tiles[2] - 1), num=num_tiles[2])
    '''
    
    x_starts = generate_tile_starts(min_coords[0], max_coords[0], tile_size, min_tile_size=5)
    y_starts = generate_tile_starts(min_coords[1], max_coords[1], tile_size, min_tile_size=5)
    z_starts = generate_tile_starts(min_coords[2], max_coords[2], tile_size, min_tile_size=5)


    ''' '''
    #x_starts = np.arange(min_coords[0], max_coords[0], tile_size)
    #y_starts = np.arange(min_coords[1], max_coords[1], tile_size)
    #z_starts = np.arange(min_coords[2], max_coords[2], tile_size)

    reconstructed_tiles = [] # List of numpy arrays that contain reconstructed coordinates of each tile
    tile_colors = []

    for x0 in x_starts:
        for y0 in y_starts:
            for z0 in z_starts:
                # Starting point of each tile
                origin = np.array([x0, y0, z0])
                bounds_min = origin
                bounds_max = origin + tile_size
                
                # Makes boolean matrix that shows indices where the point cloud's x,y, and z coords
                # are within the tile bounds.
                # Then, along each coordinate point, use np.all to see which coordinate has all 3 "Trues", meaning
                # that coordinate will be the one selected in tile_points
                # Check if entire point cloud is within the tile bounds (create boolean mask)
                in_tile_mask = np.all((point_cloud >= bounds_min) & (point_cloud < bounds_max), axis=1)
                tile_points = point_cloud[in_tile_mask] # Select only the points inside the current tile by applying the boolean mask.

                if tile_points.shape[0] == 0:
                    continue

                # Mock reconstruction
                #reconstructed_tile = tile_points + np.random.normal(scale=0.01, size=tile_points.shape)
                #reconstructed_tile = tile_points
                
                # Reconstruction
                reconstructed_tile, solvertime = reconstruct_tile(tile_points)
                #print(reconstructed_tile)
                #arr_solvertime = np.append(arr_solvertime, solvertime)
                #sum_solvertime = np.sum(solvertime)
                
                
                
                '''
                    1. Create Point CLoud Object
                    2. Set LAS coords from tile_points in the PC object
                    3. Do the whole reconstruction procedure (input reconstruction parameters)
                    4. Without showing/plotting the PC tile or outputting reconstruction errors,
                       output only the reconstructed PC coords as (m x 3) ndarray
                    FOR NOW: Ignore individual results (errors, solvertimes) of each tile. Only obtain them as a whole at the end.
                '''
                
                '''
                    1. pc = CSPC.CSPCdwt()
                    2. pc.setManualCoords(tile_points[:,0], tile_points[:,1], tile_points[:,2])
                    3. thresholding, measurement, reconstruction
                    4. reconstructed_tile = np.stack((pc.x_r, pc.y_r, pc.z_r), axis=1)
                    FOR NOW: Ignore individual results (errors, solvertimes) of each tile. Only obtain them as a whole at the end.
                '''
                
                reconstructed_tiles.append(reconstructed_tile)

                # Assign a random color to all points in each tile
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
    #ax2.set_xlim(min(combined_cloud[:,0]), max(combined_cloud[:,0]))
    #ax2.set_ylim(min(combined_cloud[:,1]), max(combined_cloud[:,1]))
    #ax2.set_zlim(min(combined_cloud[:,2]), max(combined_cloud[:,2]))
    ax2.set_xlim(min(point_cloud[:,0]), max(point_cloud[:,0]))
    ax2.set_ylim(min(point_cloud[:,1]), max(point_cloud[:,1]))
    ax2.set_zlim(min(point_cloud[:,2]), max(point_cloud[:,2]))


    plt.tight_layout()
    plt.show()