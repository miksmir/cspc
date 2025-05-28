# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 11:29:51 2025

@author: misha
"""

from CSPointCloud import CSPCdwt, CSPCdct, CSPCdft
import CSPointCloud as CSPC

def runCSPCdwt(inputpath: str, path: str, lasfile: str, pcname: str, sparsity_val: int, cs_ratio: float, measurement_type: str, wvlt: str, ds_type: str, parallel = True):
    '''
    This function runs the script for CSPC DWT reconstruction.
    
    Parameters:
        - inputpath: Path + Filename to .las file that is being reconstructed
        - path: Path to directory that will generate outputs files of reconstruction
        - lasfile: Name of output files of reconstruction that will be generated. ***MUST end in .las extension!***
        - pcname: Name of point cloud for plot and metadata labeling purposes
        - sparsity_val: Percentage of signal that will be sparse (have zero values). Example: If sparsity_val = 10, 10% of signal will have zero values.
        - cs_ratio: Compression ratio percentage. Example: 25% compression means 75% of points are removed.
        - measurement_type: Type of measurement matrix. (gaussian, gaussian_normal, bernoulli_standard, bernoulli_symmetric, gaussian2)
        - wvlt: Type of discrete wavelet used for DWT reconstruction in pywavelets (haar, db2, coif1, etc)
        - ds_type: Type of downsampling that was done on input signal. (uniform, voxel)
        
    '''
    
    ''' NYCOpenData DWT Example '''
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdwt()
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=CSPC.pclength(inputpath), cs_ratio=cs_ratio, measurement_type=measurement_type, basis='DWT', wvlt=wvlt, ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------
    
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    fcx_th, thx = CSPC.applyThresholding(fcx, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep (nonzer0)
    fcy_th, thy = CSPC.applyThresholding(fcy, percentile=sparsity_percentile, type_th ='hard')
    fcz_th, thz = CSPC.applyThresholding(fcz, percentile=sparsity_percentile, type_th ='hard')
            #TODO: Try performing thresholding after measurement
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
        solvertime = nycpc.reconstructCVXPY_ray(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
    else:
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
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
    

def runCSPCdct(inputpath: str, path: str, lasfile: str, pcname: str, sparsity_val: int, cs_ratio: float, measurement_type: str, ds_type: str, parallel = True):
    '''
    This function runs the script for CSPC DCT reconstruction.
    
    Parameters:
        - inputpath: Path + Filename to .las file that is being reconstructed
        - path: Path to directory that will generate outputs files of reconstruction
        - lasfile: Name of output files of reconstruction that will be generated. ***MUST end in .las extension!***
        - pcname: Name of point cloud for plot and metadata labeling purposes
        - sparsity_val: Percentage of signal that will be sparse (have zero values). Example: If sparsity_val = 10, 10% of signal will have zero values.
        - cs_ratio: Compression ratio percentage. Example: 25% compression means 75% of points are removed.
        - measurement_type: Type of measurement matrix. (gaussian, gaussian_normal, bernoulli_standard, bernoulli_symmetric, gaussian2)
        - ds_type: Type of downsampling that was done on input signal. (uniform, voxel)
        
    '''
    ''' NYCOpenData DCT Example '''
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdct()
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=CSPC.pclength(inputpath), cs_ratio=cs_ratio, measurement_type=measurement_type, basis='DCT', wvlt='', ds_type=ds_type, sparsity=sparsity_val)
    
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
    if(parallel):
        solvertime = nycpc.reconstructCVXPY_ray(m_x, m_y, m_z, phi)
    else:
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
    

def runCSPCdct2(inputpath: str, path: str, lasfile: str, pcname: str, sparsity_val: int, cs_ratio: float, measurement_type: str, ds_type: str):
    '''
    This function runs the script for CSPC DCT reconstruction.
    
    Parameters:
        - inputpath: Path + Filename to .las file that is being reconstructed
        - path: Path to directory that will generate outputs files of reconstruction
        - lasfile: Name of output files of reconstruction that will be generated. ***MUST end in .las extension!***
        - pcname: Name of point cloud for plot and metadata labeling purposes
        - sparsity_val: Percentage of signal that will be sparse (have zero values). Example: If sparsity_val = 10, 10% of signal will have zero values.
        - cs_ratio: Compression ratio percentage. Example: 25% compression means 75% of points are removed.
        - measurement_type: Type of measurement matrix. (gaussian, gaussian_normal, bernoulli_standard, bernoulli_symmetric, gaussian2)
        - ds_type: Type of downsampling that was done on input signal. (uniform, voxel)
        
    '''
    ''' NYCOpenData DCT Example '''
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdct()
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=CSPC.pclength(inputpath), cs_ratio=cs_ratio, measurement_type=measurement_type, basis='DCT', wvlt='', ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DCT
    cx, cy, cz = nycpc.transformDCT1D()
    # Calculate sparsity of each basis-transformed coordinates
    s, sn = CSPC.calculateSparsity(cx, cy, cz)
        #s, sn = CSPC.calculateSparsity(cx, cy, cz)
    # Create measurement matrix
    n_coeffs = len(cx)
        #n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
        #m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=cx, y_flat_coeffs=cy, z_flat_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    # IF NEEDED: Perform thresholding
    sparsity_percentile = 100 - sparsity_val # How much of the signal is left with nonzero coefficients
    cx_th, thx = CSPC.applyThresholding(m_x, percentile=sparsity_percentile, type_th ='hard') #percentile is the percentage of coefficients to keep
    cy_th, thy = CSPC.applyThresholding(m_y, percentile=sparsity_percentile, type_th ='hard')
    cz_th, thz = CSPC.applyThresholding(m_z, percentile=sparsity_percentile, type_th ='hard')
    solvertime = nycpc.reconstructCVXPY(cx_th, cy_th, cz_th, phi)
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

def runCSPCdft(inputpath: str, path: str, lasfile: str, pcname: str, sparsity_val: int, cs_ratio: float, measurement_type: str, ds_type: str):
    '''
    This function runs the script for CSPC DFT reconstruction.
    
    Parameters:
        - inputpath: Path + Filename to .las file that is being reconstructed
        - path: Path to directory that will generate outputs files of reconstruction
        - lasfile: Name of output files of reconstruction that will be generated. ***MUST end in .las extension!***
        - pcname: Name of point cloud for plot and metadata labeling purposes
        - sparsity_val: Percentage of signal that will be sparse (have zero values). Example: If sparsity_val = 10, 10% of signal will have zero values.
        - cs_ratio: Compression ratio percentage. Example: 25% compression means 75% of points are removed.
        - measurement_type: Type of measurement matrix. (gaussian, gaussian_normal, bernoulli_standard, bernoulli_symmetric, gaussian2)
        - ds_type: Type of downsampling that was done on input signal. (uniform, voxel)
        
    '''
    ''' NYCOpenData DFT Example '''
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdft()
    
    outputpath, plot_title, metadata = CSPC.setupParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=CSPC.pclength(inputpath), cs_ratio=cs_ratio, measurement_type=measurement_type, basis='DFT', wvlt='', ds_type=ds_type, sparsity=sparsity_val)
    
    # ---------------------

    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    #nycpc.showPC()
    # 1D DFT
    cx, cy, cz = nycpc.transformDFT1D()
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
    


