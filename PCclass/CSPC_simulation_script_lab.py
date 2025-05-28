# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:54:21 2025

@author: Mikhail
"""

import CSPointCloud as CSPC
import numpy as np
import os

if __name__ == "__main__":
    
    
    
    '''
    # Path of downsampled point cloud being used as input
    inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_11230ds_uniform.las'
    #inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters
    path = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/' # Output path
    #lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # Original name of .las file (not downsampled file)
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Custom named original output file
    pcname = 'MtMarcy'
    cs_ratio = 0.05
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'haar'
    ds_type = 'uniform'
    iterations = 1 # Simulation iterations
    num_points = CSPC.pclength(inputpath)
    
    outputpath, plot_title, metadata = CSPC.setupSimulationParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, i=iterations)
    
    
    # Simulation output arrays
    arr_l2norm = np.zeros(iterations)
    arr_MSE = np.zeros(iterations)
    arr_RMSE = np.zeros(iterations)
    arr_MAE = np.zeros(iterations)
    arr_solvertime = np.zeros(iterations) 
    
    # Instantiate CSPointCloud object
    nycpc = CSPC.CSPCdwt()
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # For measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    
    for i in range(0, iterations):
        # Create measurement matrix
        phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
        # Measure (subsample) each of the point cloud dimensions (y = Cx)
        m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
        # Get reconstruction errors as a dictionary
        reconstructionErrors = nycpc.calculateReconstructionError()
        # Calculate sparsity of each basis-transformed coordinates
        s = CSPC.calculateSparsity(fcx, fcy, fcz)
        # Recording simulation outputs
        arr_l2norm[i] = reconstructionErrors['l2norm']
        arr_MSE[i] = reconstructionErrors['MSE']
        arr_RMSE[i] = reconstructionErrors['RMSE']
        arr_MAE[i] = reconstructionErrors['MAE']
        arr_solvertime[i] = solvertime
        print(f'Iteration #{i}')
    #CSPC.exportSimulationInfo(outputpath, metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    CSPC.exportSimulationInfo('/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/MtMarcy_u_5865088400_2015_DWT_Haar_05Gaussian_of_11230ds_uniform_1simulations.txt', metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    
    print('Simulation Done')
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Path of downsampled point cloud being used as input
    inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_11230ds_uniform.las'
    #inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters
    path = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/' # Output path
    #lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # Original name of .las file (not downsampled file)
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Custom named original output file
    pcname = 'MtMarcy'
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'haar'
    ds_type = 'uniform'
    iterations = 100 # Simulation iterations
    num_points = CSPC.pclength(inputpath)
    
    outputpath, plot_title, metadata = CSPC.setupSimulationParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, i=iterations)
    
    
    # Simulation output arrays
    arr_l2norm = np.zeros(iterations)
    arr_MSE = np.zeros(iterations)
    arr_RMSE = np.zeros(iterations)
    arr_MAE = np.zeros(iterations)
    arr_solvertime = np.zeros(iterations) 
    
    # Instantiate CSPointCloud object
    nycpc = CSPC.CSPCdwt()
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # For measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    
    for i in range(0, iterations):
        # Create measurement matrix
        phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
        # Measure (subsample) each of the point cloud dimensions (y = Cx)
        m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
        # Get reconstruction errors as a dictionary
        reconstructionErrors = nycpc.calculateReconstructionError()
        # Calculate sparsity of each basis-transformed coordinates
        s = CSPC.calculateSparsity(fcx, fcy, fcz)
        # Recording simulation outputs
        arr_l2norm[i] = reconstructionErrors['l2norm']
        arr_MSE[i] = reconstructionErrors['MSE']
        arr_RMSE[i] = reconstructionErrors['RMSE']
        arr_MAE[i] = reconstructionErrors['MAE']
        arr_solvertime[i] = solvertime
        print(f'Iteration #{i}')
    #CSPC.exportSimulationInfo(outputpath, metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    CSPC.exportSimulationInfo('/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/MtMarcy_u_5865088400_2015_DWT_Haar_05Gaussian_of_11230ds_uniform_1simulations.txt', metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    print('Simulation Done')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Path of downsampled point cloud being used as input
    inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_11230ds_uniform.las'
    #inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters
    path = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/' # Output path
    #lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # Original name of .las file (not downsampled file)
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Custom named original output file
    pcname = 'MtMarcy'
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'db2'
    ds_type = 'uniform'
    iterations = 100 # Simulation iterations
    num_points = CSPC.pclength(inputpath)
    
    outputpath, plot_title, metadata = CSPC.setupSimulationParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, i=iterations)
    
    
    # Simulation output arrays
    arr_l2norm = np.zeros(iterations)
    arr_MSE = np.zeros(iterations)
    arr_RMSE = np.zeros(iterations)
    arr_MAE = np.zeros(iterations)
    arr_solvertime = np.zeros(iterations) 
    
    # Instantiate CSPointCloud object
    nycpc = CSPC.CSPCdwt()
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # For measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    
    for i in range(0, iterations):
        # Create measurement matrix
        phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
        # Measure (subsample) each of the point cloud dimensions (y = Cx)
        m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
        # Get reconstruction errors as a dictionary
        reconstructionErrors = nycpc.calculateReconstructionError()
        # Calculate sparsity of each basis-transformed coordinates
        s = CSPC.calculateSparsity(fcx, fcy, fcz)
        # Recording simulation outputs
        arr_l2norm[i] = reconstructionErrors['l2norm']
        arr_MSE[i] = reconstructionErrors['MSE']
        arr_RMSE[i] = reconstructionErrors['RMSE']
        arr_MAE[i] = reconstructionErrors['MAE']
        arr_solvertime[i] = solvertime
        print(f'Iteration #{i}')
    #CSPC.exportSimulationInfo(outputpath, metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    CSPC.exportSimulationInfo('/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/MtMarcy_u_5865088400_2015_DWT_Db2_05Gaussian_of_11230ds_uniform_1simulations.txt', metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    print('Simulation Done')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Path of downsampled point cloud being used as input
    inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_11230ds_uniform.las'
    #inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters
    path = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/' # Output path
    #lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # Original name of .las file (not downsampled file)
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Custom named original output file
    pcname = 'MtMarcy'
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'sym2'
    ds_type = 'uniform'
    iterations = 100 # Simulation iterations
    num_points = CSPC.pclength(inputpath)
    
    outputpath, plot_title, metadata = CSPC.setupSimulationParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, i=iterations)
    
    
    # Simulation output arrays
    arr_l2norm = np.zeros(iterations)
    arr_MSE = np.zeros(iterations)
    arr_RMSE = np.zeros(iterations)
    arr_MAE = np.zeros(iterations)
    arr_solvertime = np.zeros(iterations) 
    
    # Instantiate CSPointCloud object
    nycpc = CSPC.CSPCdwt()
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # For measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    
    for i in range(0, iterations):
        # Create measurement matrix
        phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
        # Measure (subsample) each of the point cloud dimensions (y = Cx)
        m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
        # Get reconstruction errors as a dictionary
        reconstructionErrors = nycpc.calculateReconstructionError()
        # Calculate sparsity of each basis-transformed coordinates
        s = CSPC.calculateSparsity(fcx, fcy, fcz)
        # Recording simulation outputs
        arr_l2norm[i] = reconstructionErrors['l2norm']
        arr_MSE[i] = reconstructionErrors['MSE']
        arr_RMSE[i] = reconstructionErrors['RMSE']
        arr_MAE[i] = reconstructionErrors['MAE']
        arr_solvertime[i] = solvertime
        print(f'Iteration #{i}')
    #CSPC.exportSimulationInfo(outputpath, metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    CSPC.exportSimulationInfo('/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/MtMarcy_u_5865088400_2015_DWT_Sym2_05Gaussian_of_11230ds_uniform_1simulations.txt', metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    print('Simulation Done')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Path of downsampled point cloud being used as input
    inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_11230ds_uniform.las'
    #inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters
    path = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/' # Output path
    #lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # Original name of .las file (not downsampled file)
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Custom named original output file
    pcname = 'MtMarcy'
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'bior1.1'
    ds_type = 'uniform'
    iterations = 100 # Simulation iterations
    num_points = CSPC.pclength(inputpath)
    
    outputpath, plot_title, metadata = CSPC.setupSimulationParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, i=iterations)
    
    
    # Simulation output arrays
    arr_l2norm = np.zeros(iterations)
    arr_MSE = np.zeros(iterations)
    arr_RMSE = np.zeros(iterations)
    arr_MAE = np.zeros(iterations)
    arr_solvertime = np.zeros(iterations) 
    
    # Instantiate CSPointCloud object
    nycpc = CSPC.CSPCdwt()
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # For measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    
    for i in range(0, iterations):
        # Create measurement matrix
        phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
        # Measure (subsample) each of the point cloud dimensions (y = Cx)
        m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
        # Get reconstruction errors as a dictionary
        reconstructionErrors = nycpc.calculateReconstructionError()
        # Calculate sparsity of each basis-transformed coordinates
        s = CSPC.calculateSparsity(fcx, fcy, fcz)
        # Recording simulation outputs
        arr_l2norm[i] = reconstructionErrors['l2norm']
        arr_MSE[i] = reconstructionErrors['MSE']
        arr_RMSE[i] = reconstructionErrors['RMSE']
        arr_MAE[i] = reconstructionErrors['MAE']
        arr_solvertime[i] = solvertime
        print(f'Iteration #{i}')
    #CSPC.exportSimulationInfo(outputpath, metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    CSPC.exportSimulationInfo('/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/MtMarcy_u_5865088400_2015_DWT_bior1.1_05Gaussian_of_11230ds_uniform_1simulations.txt', metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    print('Simulation Done')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Path of downsampled point cloud being used as input
    inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_11230ds_uniform.las'
    #inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters
    path = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/' # Output path
    #lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # Original name of .las file (not downsampled file)
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Custom named original output file
    pcname = 'MtMarcy'
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'rbio1.1'
    ds_type = 'uniform'
    iterations = 100 # Simulation iterations
    num_points = CSPC.pclength(inputpath)
    
    outputpath, plot_title, metadata = CSPC.setupSimulationParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, i=iterations)
    
    
    # Simulation output arrays
    arr_l2norm = np.zeros(iterations)
    arr_MSE = np.zeros(iterations)
    arr_RMSE = np.zeros(iterations)
    arr_MAE = np.zeros(iterations)
    arr_solvertime = np.zeros(iterations) 
    
    # Instantiate CSPointCloud object
    nycpc = CSPC.CSPCdwt()
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # For measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    
    for i in range(0, iterations):
        # Create measurement matrix
        phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
        # Measure (subsample) each of the point cloud dimensions (y = Cx)
        m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
        # Get reconstruction errors as a dictionary
        reconstructionErrors = nycpc.calculateReconstructionError()
        # Calculate sparsity of each basis-transformed coordinates
        s = CSPC.calculateSparsity(fcx, fcy, fcz)
        # Recording simulation outputs
        arr_l2norm[i] = reconstructionErrors['l2norm']
        arr_MSE[i] = reconstructionErrors['MSE']
        arr_RMSE[i] = reconstructionErrors['RMSE']
        arr_MAE[i] = reconstructionErrors['MAE']
        arr_solvertime[i] = solvertime
        print(f'Iteration #{i}')
    #CSPC.exportSimulationInfo(outputpath, metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    CSPC.exportSimulationInfo('/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/MtMarcy_u_5865088400_2015_DWT_rbio1.1_05Gaussian_of_11230ds_uniform_1simulations.txt', metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    print('Simulation Done')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Path of downsampled point cloud being used as input
    inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_11230ds_uniform.las'
    #inputpath = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'

    # Parameters
    path = '/home/network-lab/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/' # Output path
    #lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # Original name of .las file (not downsampled file)
    lasfile = 'MtMarcy_u_5865088400_2015.las' # Custom named original output file
    pcname = 'MtMarcy'
    cs_ratio = 0.5
    measurement_type = 'gaussian'
    basis = 'DWT'
    wvlt = 'coif1'
    ds_type = 'uniform'
    iterations = 100 # Simulation iterations
    num_points = CSPC.pclength(inputpath)
    
    outputpath, plot_title, metadata = CSPC.setupSimulationParameters(path=path, lasfile=lasfile, pcname=pcname, num_points=num_points, cs_ratio=cs_ratio, measurement_type=measurement_type, basis=basis, wvlt=wvlt, ds_type=ds_type, i=iterations)
    
    
    # Simulation output arrays
    arr_l2norm = np.zeros(iterations)
    arr_MSE = np.zeros(iterations)
    arr_RMSE = np.zeros(iterations)
    arr_MAE = np.zeros(iterations)
    arr_solvertime = np.zeros(iterations) 
    
    # Instantiate CSPointCloud object
    nycpc = CSPC.CSPCdwt()
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # For measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    
    for i in range(0, iterations):
        # Create measurement matrix
        phi = CSPC.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
        # Measure (subsample) each of the point cloud dimensions (y = Cx)
        m_x, m_y, m_z = CSPC.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
        # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
        solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
        # Get reconstruction errors as a dictionary
        reconstructionErrors = nycpc.calculateReconstructionError()
        # Calculate sparsity of each basis-transformed coordinates
        s = CSPC.calculateSparsity(fcx, fcy, fcz)
        # Recording simulation outputs
        arr_l2norm[i] = reconstructionErrors['l2norm']
        arr_MSE[i] = reconstructionErrors['MSE']
        arr_RMSE[i] = reconstructionErrors['RMSE']
        arr_MAE[i] = reconstructionErrors['MAE']
        arr_solvertime[i] = solvertime
        print(f'Iteration #{i}')
    #CSPC.exportSimulationInfo(outputpath, metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    CSPC.exportSimulationInfo('/home/network-lab/Documents/Thesis_CS/Point_Clouds/Point_Cloud_Outputs/reconstruction/NYCOpenData/simulation/MtMarcy_u_5865088400_2015_DWT_Coif1_05Gaussian_of_11230ds_uniform_1simulations.txt', metadata, arr_l2norm, arr_MSE, arr_RMSE, arr_MAE, arr_solvertime)
    print('Simulation Done')
    