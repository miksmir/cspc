# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 18:55:59 2025

@author: misha
"""
import CSPointCloud as CSPC
from CSPointCloud import CSPCdct


if __name__ == "__main__":
    ''' NYCOpenData DCT Example '''
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdct()
    
    # Specify input path
    #inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las'
    inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las'
    #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_6712ds_uniform.las'
    #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Dumbo_985195_LidarClassifiedPointCloud_4972ds_uniform.las'
    #inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las'
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 10 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.75
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
        #s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 20 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.75
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 30 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.75
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 10 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 20 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 30 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 10 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.25
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 20 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.25
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Parameters ----------  
        #path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    path = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding' # Output path
    sparsity_val = 30 # Percentage of signal that will be sparse (have zero values)
    #lasfile = f'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    lasfile = f'MtMarcy_u_5865088400_2015_{sparsity_val}thresholded.las' # Original name of .las file (not downsampled file)
    pcname = 'Mount Marcy'
    num_points = CSPC.pclength(inputpath)
    cs_ratio = 0.25
    measurement_type = 'gaussian'
    basis = 'DCT'
    wvlt = ''
    ds_type = 'uniform' # 'uniform', 'voxel' or 'none'
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(cx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(cy, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(cz, percentile=sparsity_percentile, type_th ='hard')
    # Calculate sparsity of each basis-transformed coordinates
    #s, sn = CSPC.calculateSparsity(cx_th, cy_th, cz_th)
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx_th)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx_th, y_flat_coeffs=cy_th, z_flat_coeffs=cz_th)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    #solvertime, xs, ys, zs = nycpc.reconstructCosamp(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt, sx=sn['x'], sy=sn['y'], sz=sn['z'])
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
        #nycpc.showReconstructedPC()
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