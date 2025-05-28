# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:00:29 2025

@author: Mikhail
"""

import CSPointCloud as CSPC
from CSPointCloud import CSPCdwt, CSPCdct, CSPCdft


if __name__ == "__main__":
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdft()
    
    # Specify input path
    inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5182ds_uniform.las'
    #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_6712ds_uniform.las'
    #inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_LidarClassifiedPointCloud_6658ds_uniform.las'

    
    # Parameters ----------  
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData'
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Original name of .las file (not downsampled file)
    pcname = 'Mt. Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DFT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type)
    
    # ---------------------
    
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDFT1D()
    # IF NEEDED: Perform thresholding
    #fcx = CSPC.applyThresholding(fcx, threshold=7e-2, type='hard') #1e-14
    #fcy = CSPC.applyThresholding(fcy, threshold=7e-2, type='hard')
    #fcz = CSPC.applyThresholding(fcz, threshold=7e-2, type='hard')
    # Calculate sparsity of each basis-transformed coordinates
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
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