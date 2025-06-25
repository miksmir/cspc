# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:50:47 2025

@author: Mikhail
"""

import CSPointCloud as CSPC
from CSPointCloud import CSPCdwt, CSPCdct, CSPCdft


if __name__ == "__main__":
    
    ''' NYCOpenData DWT Example '''
    
    # For TESTING ON HELIX AND BUnNY:
    '''
    # Instantiate CSPointCloud object
    nycpc = CSPCdwt()
    
    # Specify input path
    #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'
    inputpath ='D:\Documents\Thesis_CS\Point_Cloud_Outputs\downsampled/bunny_003ds_uniform.las'
    
    # Parameters ----------  
    path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction' # Output path
    #lasfile = 'helix.las' # Original name of .las file (not downsampled file)
    lasfile = 'bunny.las'
    #pcname = 'Helix'
    pcname = 'Bunny'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'db2'
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type)
    '''
    
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdwt()
    
    # Specify input path
    #inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las'
    #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_6712ds_uniform.las'
    #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Dumbo_985195_LidarClassifiedPointCloud_4972ds_uniform.las'
    inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las'
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData' # Output path
    lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las' # Original name of .las file (not downsampled file)
    pcname = 'Park Slope'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'coif1'
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type)
    
    # ---------------------
    
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        #fcx_th = CSPC.applyThresholding(fcx, threshold=10, type='soft') #1e-14
        #fcy_th = CSPC.applyThresholding(fcy, threshold=10, type='soft')
        #fcz_th = CSPC.applyThresholding(fcz, threshold=10, type='soft')
        #fcx_th = CSPC.applyThresholding(fcx, threshold=4.e-3, type='hard') #1e-14
        #fcy_th = CSPC.applyThresholding(fcy, threshold=4.e-3, type='hard')
        #fcz_th = CSPC.applyThresholding(fcz, threshold=4.e-3, type='hard')
    # Calculate sparsity of each basis-transformed coordinates
    s, sn = CSPC.calculateSparsity(fcx, fcy, fcz)
        #s, sn = CSPC.calculateSparsity(fcx_th, fcy_th, fcz_th)
    # Create measurement matrix
        #n_coeffs = len(fcx_th)
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx_th, y_flat_coeffs=fcy_th, z_flat_coeffs=fcz_th)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    nycpc.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    nycpc.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = nycpc.calculateReconstructionError()
    # Export error and solver time to a .txt file
    CSPC.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    nycpc.plotPCs(main_title=plot_title, e_angle=35, a_angle=-215,outputfile=outputpath, fileformat='pdf', colormap='BGYR')
    CSPC.plotContourPC(nycpc.x,nycpc.y,nycpc.z, title='Original Point Cloud', colormap='BGYR')
    CSPC.plotContourPC(nycpc.x_r,nycpc.y_r,nycpc.z_r, title='Reconstructed Point Cloud', colormap='BGYR')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    nycpc.exportCoords(outputfileoriginal=inputpath, outputfilereconstructed=outputpath, outputformat='npy', exportchoice='both')
    