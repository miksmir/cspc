# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:22:32 2025

@author: misha
"""

import CSPointCloud as CSPC
from CSPointCloud import CSPCdwt, CSPCdct, CSPCdft


if __name__ == "__main__":
    
    ''' NYCOpenData DWT Example '''
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdct()
    
    # Specify input path
    #inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_LidarClassifiedPointCloud_6658ds_uniform.las'
    inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters ----------  
    #path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction' # Output path
    #lasfile = 'Columbus_Circle_987217_LidarClassifiedPointCloud.las' # Original name of .las file (not downsampled file)
    lasfile = 'helix.las' # Original name of .las file (not downsampled file)
    #pcname = 'Columbus Circle'
    pcname = 'Helix'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type)
    
    # ---------------------
    '''
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDCT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPC.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPC.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPC.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    nycpc.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    nycpc.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = nycpc.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPC.calculateSparsity(fcx, fcy, fcz)
    # Export error and solver time to a .txt file
    CSPC.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    nycpc.plotPCs(main_title=plot_title, e_angle=35, a_angle=-215,outputfile=outputpath, fileformat='pdf', colormap='BGYR')
    CSPC.plotContourPC(nycpc.x,nycpc.y,nycpc.z, title='Original Point Cloud', colormap='BGYR')
    CSPC.plotContourPC(nycpc.x_r,nycpc.y_r,nycpc.z_r, title='Reconstructed Point Cloud', colormap='BGYR')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    nycpc.exportCoords(outputfileoriginal=inputpath, outputfilereconstructed=outputpath, outputformat='npy', exportchoice='both')
    '''
    
    
    
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    nycpc.showPC()
    # IF NEEDED: Perform downsampling:
        # # ds_percentage = 0.5
        # # dsrate = int(1/ds_percentage)
        # pc1dct.performDownsample(dsrate)
    # Multilevel 1D DCT
    coeffs_x, coeffs_y, coeffs_z = nycpc.transformDCT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPCdct.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPCdct.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPCdct.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(coeffs_x)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type='gaussian')
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=coeffs_x, y_flat_coeffs=coeffs_y, z_flat_coeffs=coeffs_z)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DCT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi, norm='ortho')
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    nycpc.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    nycpc.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = nycpc.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPC.calculateSparsity(coeffs_x, coeffs_y, coeffs_z)
    # Export error and solver time to a .txt file
    CSPCdct.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    nycpc.plotPCs(main_title=plot_title, outputfile=outputpath, fileformat='pdf')
    CSPC.plotContourPC(nycpc.x,nycpc.y,nycpc.z, title='Original Point Cloud', colormap='BGYR')
    CSPC.plotContourPC(nycpc.x_r,nycpc.y_r,nycpc.z_r, title='Reconstructed Point Cloud', colormap='BGYR')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
        # pc1.exportCoords('D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_original_helix', 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_recon_helix_DWT_025m_of_100ds', 'npy')
    nycpc.exportCoords(outputfileoriginal=inputpath, outputfilereconstructed=outputpath, outputformat='npy', exportchoice='both')
