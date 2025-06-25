from CSPointCloud import CSPCdwt, CSPCdct, CSPCdft
import numpy as np
import pywt

if __name__ == "__main__":

    
    ''' HELIX --------------------------------------------------------------------------------------------------- '''

    ''' Helix DWT Example '''
    
    """
    
    # Instantiate CSPointCloud object
    pc1 = CSPCdwt() 
    
    # Parameters ----------    
    # Specify input path and output path
    inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'
    outputpath = CSPCdwt.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/helix', 'helix_reconstructed', '_DWT_sym4_050m_of_100ds.las')
    plot_title = "50% Compression of 500-Point Helix PC Using sym4 DWT"
    metadata = "50% CS Ratio | 500 points (Not Downsampled) | Helix PC | Gaussian Measurement | sym4 DWT | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
    # Set compression ratio
    cs_ratio = 0.50
    # Specify wavelet type (for PyWavelets library)
    # To find available types of wavelets, run pywt.wavelist()
    # The only ones that have varying orders are bior, cgau, coif, db, gaus, rbio, and sym.
    wvlt = 'sym4'
    # ---------------------
    
    # Import .las file
    pc1.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    pc1.showPC()
    # IF NEEDED: Perform downsampling:
        # # ds_percentage = 0.5
        # # dsrate = int(1/ds_percentage)
        # pc1.performDownsample(dsrate)
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = pc1.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPCdwt.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPCdwt.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPCdwt.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdwt.generateMeasurementMatrix(m, n_coeffs, type='gaussian')
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdwt.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = pc1.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    pc1.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    pc1.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = pc1.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdwt.calculateSparsity(fcx, fcy, fcz)
    # Export error and solver time to a .txt file
    CSPCdwt.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    pc1.plotPCs(main_title=plot_title, outputfile=outputpath, fileformat='pdf')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
        # pc1.exportCoords('D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_original_helix', 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_recon_helix_DWT_025m_of_100ds', 'npy')
    
    """
    
    
    
    
    
    ''' Helix DCT Example '''
    """
    # Instantiate CSPointCloud object
    pc1dct = CSPCdct() 
    
    # Parameters ----------    
    # Specify input path and output path
    inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'
    outputpath = CSPCdct.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/helix', 'helix_reconstructed', '_DCT_010m_of_100ds.las')
    plot_title = "10% Compression of 500-Point Helix PC Using DCT"
    metadata = "10% CS Ratio | 500 points (Not Downsampled) | Helix PC | Gaussian Measurement | DCT | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
    # Set compression ratio
    cs_ratio = 0.10
    # ---------------------
    
    # Import .las file
    pc1dct.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    pc1dct.showPC()
    # IF NEEDED: Perform downsampling:
        # # ds_percentage = 0.5
        # # dsrate = int(1/ds_percentage)
        # pc1dct.performDownsample(dsrate)
    # Multilevel 1D DCT
    coeffs_x, coeffs_y, coeffs_z = pc1dct.transformDCT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPCdct.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPCdct.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPCdct.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(coeffs_x)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdct.generateMeasurementMatrix(m, n_coeffs, type='gaussian')
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdct.measure1D(Phi = phi, x_coeffs=coeffs_x, y_coeffs=coeffs_y, z_coeffs=coeffs_z)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DCT
    solvertime = pc1dct.reconstructCVXPY(m_x, m_y, m_z, phi, norm='ortho')
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    pc1dct.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    pc1dct.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = pc1dct.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdct.calculateSparsity(coeffs_x, coeffs_y, coeffs_z)
    # Export error and solver time to a .txt file
    CSPCdct.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    pc1dct.plotPCs(main_title=plot_title, outputfile=outputpath, fileformat='pdf')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
        # pc1.exportCoords('D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_original_helix', 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_recon_helix_DWT_025m_of_100ds', 'npy')

    """
    
    
    
    
    
    
    ''' Helix DFT Example '''
     
    """
    # Instantiate CSPointCloud object
    pc1dft = CSPCdft() 
 
     # Parameters ----------    
     # Specify input path and output path
    inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/helix.las'
    outputpath = CSPCdwt.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/helix', 'helix_reconstructed', '_DFT_075m_of_100ds.las')
    plot_title = "75% Compression of 500-Point Helix PC Using DFT"
    metadata = "75% CS Ratio | 500 points (Not Downsampled) | Helix PC | Gaussian Measurement | DFT | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
     # Set compression ratio
    cs_ratio = 0.75
     # ---------------------
     
     # Import .las file
    pc1dft.setLasCoords(inputpath)
     # Show Open3D plot of point cloud
    pc1dft.showPC()
     # IF NEEDED: Perform downsampling:
         # # ds_percentage = 0.5
         # # dsrate = int(1/ds_percentage)
         # pc1dft.performDownsample(dsrate)
     # Multilevel 1D DCT
    coeffs_x, coeffs_y, coeffs_z = pc1dft.transformDFT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
     # IF NEEDED: Perform thresholding
         # fcx = CSPCdft.applyThresholding(fcx, threshold=1e-14)
         # fcy = CSPCdft.applyThresholding(fcy, threshold=1e-14)
         # fcz = CSPCdft.applyThresholding(fcz, threshold=1e-14)
     # Create measurement matrix
    n_coeffs = len(coeffs_x)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdft.generateMeasurementMatrix(m, n_coeffs, type='gaussian')
     # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdft.measure1D(Phi = phi, x_coeffs=coeffs_x, y_coeffs=coeffs_y, z_coeffs=coeffs_z)
     # Reconstruct each point cloud dimension via L! Minimization and transform back from DCT
    solvertime = pc1dft.reconstructCVXPY(m_x, m_y, m_z, phi)
     # IF NEEDED: Show Open3D plot of reconstructed point cloud
    pc1dft.showReconstructedPC()
     # Export reconstructed point cloud as a .las file
    pc1dft.writeReconstructedLas(path=outputpath)
     # Get reconstruction errors as a dictionary
    reconstructionErrors = pc1dft.calculateReconstructionError()
     # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdft.calculateSparsity(coeffs_x, coeffs_y, coeffs_z)
     # Export error and solver time to a .txt file
    CSPCdft.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
     # Plot and export plots of both original point cloud and reconstructed point cloud
    pc1dft.plotPCs(main_title=plot_title, outputfile=outputpath, fileformat='pdf')
     # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
         # pc1dft.exportCoords('D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_original_helix', 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_recon_helix_DWT_025m_of_100ds', 'npy')
    """
    
    
    ''' Stanford Bunny ---------------------------------------------------------------------- '''
    
    
    
    """ Creating and exporting a downsampled point cloud example """
    """
    pc_ds = CSPCdwt()
    pc_ds.setLasCoords('D:/Documents/Thesis_CS/Point_Cloud_Inputs/bun000.las')
    pc_ds.showPC()
    # Downsampling
    ds_percentage = 0.1
    dsrate = int(1/ds_percentage) # Keep every "dsrate" amount of points
    pc_ds.performDownsample(dsrate, showPC=True, dstype='uniform')
    num_points = pc_ds.x.size # Amount of resulting points after downsampling
    ds_path = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/bunny_{num_points}ds_uniform.las'
    pc_ds.writeLas(path=ds_path)
    """
    
    
    
    ''' Stanford Bunny DWT Example '''
    
    """
    # Instantiate CSPointCloud object
    pc1dwt = CSPCdwt()
    
     # Parameters ----------    
     # Specify input path and output path
         #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/bun000.las'
    inputpath = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/bunny_{num_points}ds_uniform.las'
    
     # Set compression ratio
    cs_ratio = 0.25
    # Type of measurement matrix
    measurement_type = 'gaussian'
    # Specify basis type
    basis = 'DWT'
    wvlt = 'haar'   # To find available types of wavelets, run pywt.wavelist()
    # For string interpolation
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    wvlt_str = f'{basis}_' + wvlt.capitalize() # Wavelet type string interpolation
    basis_str = wvlt.capitalize() + f' {basis}' # Wavelet type string interpolation
    measurement_type_str = measurement_type.capitalize() # Measurement type string interpolation
    # Specify output path and output names
    outputpath = CSPCdwt.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/bunny', 'bunny_reconstructed', f'_{wvlt_str}_{cs_ratio_str}m_of_{num_points}ds_uniform.las')
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point Bunny PC Using {basis_str}"
    metadata = f"{cs_percentage_str}% CS Ratio | {num_points} points (Uniform Downsampled) | Stanford Bunny PC | {measurement_type_str} Measurement | {basis_str} | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
    
    
    # ---------------------
    
    # Import .las file
    pc1dwt.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    pc1dwt.showPC()
    # IF NEEDED: Perform downsampling:
        #pc1dwt.performDownsample(dsrate)
        # Write the downsampled PC to a .las file
        #pc1dwt.writeLas(path='D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/bunny_010ds_uniform.las')
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = pc1dwt.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPCdwt.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPCdwt.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPCdwt.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdwt.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdwt.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = pc1dwt.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    pc1dwt.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    pc1dwt.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = pc1dwt.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdwt.calculateSparsity(fcx, fcy, fcz)
    # Export error and solver time to a .txt file
    CSPCdwt.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    pc1dwt.plotPCs(main_title=plot_title, outputfile=outputpath, fileformat='pdf')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    pc1dwt.exportCoords(outputpath, 'npy')  
    """
    
    
    
    ''' Stanford Bunny DCT Example '''
    
    """
    num_points = 4026
    # Instantiate CSPointCloud object
    pc1dct = CSPCdct() 
 
     # Parameters ----------    
     # Specify input path and output path
         #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/bun000.las'
    inputpath = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/bunny_{num_points}ds_uniform.las'
    
     # Set compression ratio
    cs_ratio = 0.05
    # Type of measurement matrix
    measurement_type = 'gaussian'
    # Specify basis type
    basis = 'DCT'
    # For string interpolation
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    basis_str = basis       # Wavelet type string interpolation
    measurement_type_str = measurement_type.capitalize()
    # Specify output path and output names
    outputpath = CSPCdct.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/bunny', 'bunny_reconstructed', f'_{basis_str}_{cs_ratio_str}m_of_{num_points}ds_uniform.las')
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point Bunny PC Using {basis_str}"
    metadata = f"{cs_percentage_str}% CS Ratio | {num_points} points (Uniform Downsampled) | Stanford Bunny PC | {measurement_type_str} Measurement | {basis_str} | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
     
    # ---------------------
    
     # Import .las file
    pc1dct.setLasCoords(inputpath)
     # Show Open3D plot of point cloud
    pc1dct.showPC()
   # IF NEEDED: Perform downsampling:
       #pc1dft.performDownsample(dsrate)
   # Write the downsampled PC to a .las file
        #pc1dft.writeLas(path='D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/bunny_010ds_uniform.las')
     # Multilevel 1D DCT
    coeffs_x, coeffs_y, coeffs_z = pc1dct.transformDCT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
     # IF NEEDED: Perform thresholding
         # fcx = CSPCdct.applyThresholding(fcx, threshold=1e-14)
         # fcy = CSPCdct.applyThresholding(fcy, threshold=1e-14)
         # fcz = CSPCdct.applyThresholding(fcz, threshold=1e-14)
     # Create measurement matrix
    n_coeffs = len(coeffs_x)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdct.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
     # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdct.measure1D(Phi = phi, x_coeffs=coeffs_x, y_coeffs=coeffs_y, z_coeffs=coeffs_z)
     # Reconstruct each point cloud dimension via L! Minimization and transform back from DCT
    solvertime = pc1dct.reconstructCVXPY(m_x, m_y, m_z, phi)
     # IF NEEDED: Show Open3D plot of reconstructed point cloud
    pc1dct.showReconstructedPC()
     # Export reconstructed point cloud as a .las file
    pc1dct.writeReconstructedLas(path=outputpath)
     # Get reconstruction errors as a dictionary
    reconstructionErrors = pc1dct.calculateReconstructionError()
     # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdct.calculateSparsity(coeffs_x, coeffs_y, coeffs_z)
     # Export error and solver time to a .txt file
    CSPCdct.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
     # Plot and export plots of both original point cloud and reconstructed point cloud
    pc1dct.plotPCs(main_title=plot_title, outputfile=outputpath, fileformat='pdf')
     # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    pc1dct.exportCoords(outputpath, 'npy')  
    """
   
    
    
    ''' Stanford Bunny DFT Example '''
    """
    # Instantiate CSPointCloud object
    pc1dft = CSPCdft() 
 
     # Parameters ----------    
     # Specify input path and output path
         #inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/bun000.las'
    inputpath = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/bunny_{num_points}ds_uniform.las'
    
     # Set compression ratio
    cs_ratio = 0.05
    # Type of measurement matrix
    measurement_type = 'gaussian'
    # Specify basis type
    basis = 'DFT'
    # For string interpolation
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    basis_str = basis       # Wavelet type string interpolation
    measurement_type_str = measurement_type.capitalize()
    # Specify output path and output names
    outputpath = CSPCdwt.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/bunny', 'bunny_reconstructed', f'_{basis_str}_{cs_ratio_str}m_of_{num_points}ds_uniform.las')
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point Bunny PC Using {basis_str}"
    metadata = f"{cs_percentage_str}% CS Ratio | {num_points} points (Uniform Downsampled) | Stanford Bunny PC | {measurement_type_str} Measurement | {basis_str} | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
     
    # ---------------------
    
     # Import .las file
    pc1dft.setLasCoords(inputpath)
     # Show Open3D plot of point cloud
    pc1dft.showPC()
   # IF NEEDED: Perform downsampling:
       #pc1dft.performDownsample(dsrate)
   # Write the downsampled PC to a .las file
        #pc1dft.writeLas(path='D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/bunny_010ds_uniform.las')
     # Multilevel 1D DCT
    coeffs_x, coeffs_y, coeffs_z = pc1dft.transformDFT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
     # IF NEEDED: Perform thresholding
         # fcx = CSPCdft.applyThresholding(fcx, threshold=1e-14)
         # fcy = CSPCdft.applyThresholding(fcy, threshold=1e-14)
         # fcz = CSPCdft.applyThresholding(fcz, threshold=1e-14)
     # Create measurement matrix
    n_coeffs = len(coeffs_x)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdft.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
     # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdft.measure1D(Phi = phi, x_coeffs=coeffs_x, y_coeffs=coeffs_y, z_coeffs=coeffs_z)
     # Reconstruct each point cloud dimension via L! Minimization and transform back from DCT
    solvertime = pc1dft.reconstructCVXPY(m_x, m_y, m_z, phi)
     # IF NEEDED: Show Open3D plot of reconstructed point cloud
    pc1dft.showReconstructedPC()
     # Export reconstructed point cloud as a .las file
    pc1dft.writeReconstructedLas(path=outputpath)
     # Get reconstruction errors as a dictionary
    reconstructionErrors = pc1dft.calculateReconstructionError()
     # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdft.calculateSparsity(coeffs_x, coeffs_y, coeffs_z)
     # Export error and solver time to a .txt file
    CSPCdft.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
     # Plot and export plots of both original point cloud and reconstructed point cloud
    pc1dft.plotPCs(main_title=plot_title, outputfile=outputpath, fileformat='pdf')
     # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    pc1dft.exportCoords(outputpath, 'npy')   
    """
    
    
    
    
    ''' NYC OpenData Point Cloud ---------------------------------------------------------------------- '''
    
    """ Creating and exporting a downsampled point cloud example """
    
    pc_ds = CSPCdct()
    pc_ds.setLasCoords('D:/Documents/Thesis_CS/Point_Cloud_Inputs/NYCOpenData/Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1.las')
    pc_ds.showPC()
    # Downsampling
    ds_percentage = 1e-3
    dsrate = int(1/ds_percentage) # Keep every "dsrate" amount of points
    pc_ds.performDownsample(dsrate, showPC=False, dstype='uniform')
    num_points = pc_ds.x.size # Amount of resulting points after downsampling
    ds_path = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1_{num_points}ds_uniform.las'
    pc_ds.writeLas(path=ds_path)
    
    
    
    ''' NYCOpenData DWT Example '''
    """
    # Instantiate CSPointCloud object
    nycpc = CSPCdwt()
    
    # Parameters ----------    
    # Specify input path
    inputpath = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1_{num_points}ds_uniform.las'
    # Set compression ratio
    cs_ratio = 0.75
    # Type of measurement matrix
    measurement_type = 'gaussian'
    # Specify wavelet type (for PyWavelets library)
    basis = 'DWT'
    wvlt = 'bior4.4'   # To find available types of wavelets, run pywt.wavelist()
    # For string interpolation
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    wvlt_str = f'{basis}_' + wvlt.capitalize() # Wavelet type string interpolation
    basis_str = wvlt.capitalize() + f' {basis}' # Wavelet type string interpolation
    measurement_type_str = measurement_type.capitalize()
    # Specify output path and output names
    outputpath = CSPCdwt.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData', 'Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1', f'_{wvlt_str}_{cs_ratio_str}m_of_{num_points}ds_uniform.las')
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point Columbus Circle PC Using {basis_str}"
    metadata = f"{cs_percentage_str}% CS Ratio | {num_points} points (Uniform Downsampled) | NYCOpenData Columbus Circle 987217 Segmented1 PC | {measurement_type_str} Measurement | {basis_str} | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
    
    # ---------------------
    
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    nycpc.showPC()
    # Multilevel 1D DWT
    fcx, cx, fcy, cy, fcz, cz = nycpc.transformDWT1D(wavelet=wvlt) # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPCdwt.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPCdwt.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPCdwt.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(fcx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdwt.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdwt.measure1D(Phi = phi, x_flat_coeffs=fcx, y_flat_coeffs=fcy, z_flat_coeffs=fcz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, cx, m_y, cy, m_z, cz, phi, wavelet=wvlt)
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    nycpc.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    nycpc.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = nycpc.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdwt.calculateSparsity(fcx, fcy, fcz)
    # Export error and solver time to a .txt file
    CSPCdwt.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    nycpc.plotPCs(main_title=plot_title, e_angle=35, a_angle=-215,outputfile=outputpath, fileformat='pdf')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    nycpc.exportCoords(outputpath, 'npy')
    """
    
    
    
    ''' NYCOpenData DCT Example '''
    """
    # Instantiate CSPointCloud object
    nycpc = CSPCdct()
    
    # Parameters ----------    
    # Specify input path
    inputpath = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1_{num_points}ds_uniform.las'
    # Set compression ratio
    cs_ratio = 0.5
    # Type of measurement matrix
    measurement_type = 'gaussian'
    # Specify wavelet type (for PyWavelets library)
    basis = 'DCT'
    #wvlt = 'bior4.4'   # To find available types of wavelets, run pywt.wavelist()
    # For string interpolation
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    #wvlt_str = f'{basis}_' + wvlt.capitalize() # Wavelet type string interpolation
    wvlt_str = basis
    basis_str = f' {basis}' # Wavelet type string interpolation
    measurement_type_str = measurement_type.capitalize()
    # Specify output path and output names
    outputpath = CSPCdct.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData', 'Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1', f'_{wvlt_str}_{cs_ratio_str}m_of_{num_points}ds_uniform.las')
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point Columbus Circle PC Using {basis_str}"
    metadata = f"{cs_percentage_str}% CS Ratio | {num_points} points (Uniform Downsampled) | NYCOpenData Columbus Circle 987217 Segmented1 PC | {measurement_type_str} Measurement | {basis_str} | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
    
    # ---------------------
    
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    nycpc.showPC()
    # Multilevel 1D DWT
    cx, cy, cz = nycpc.transformDCT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPCdwt.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPCdwt.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPCdwt.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdct.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdct.measure1D(Phi = phi, x_coeffs=cx, y_coeffs=cy, z_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    nycpc.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    nycpc.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = nycpc.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdct.calculateSparsity(cx, cy, cz)
    # Export error and solver time to a .txt file
    CSPCdct.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    nycpc.plotPCs(main_title=plot_title, e_angle=35, a_angle=-215,outputfile=outputpath, fileformat='pdf')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    nycpc.exportCoords(outputpath, 'npy')
    """
    
    
    
    ''' NYCOpenData DFT Example '''
    
    # Instantiate CSPointCloud object
    nycpc = CSPCdft()
    
    # Parameters ----------    
    # Specify input path
    inputpath = f'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1_{num_points}ds_uniform.las'
    # Set compression ratio
    cs_ratio = 0.5
    # Type of measurement matrix
    measurement_type = 'gaussian'
    # Specify wavelet type (for PyWavelets library)
    basis = 'DFT'
    #wvlt = 'bior4.4'   # To find available types of wavelets, run pywt.wavelist()
    # For string interpolation
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    #wvlt_str = f'{basis}_' + wvlt.capitalize() # Wavelet type string interpolation
    wvlt_str = basis
    basis_str = f' {basis}' # Wavelet type string interpolation
    measurement_type_str = measurement_type.capitalize()
    # Specify output path and output names
    outputpath = CSPCdft.specifyOutputPath('D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData', 'Columbus_Circle_987217_LidarClassifiedPointCloud_segmented1', f'_{wvlt_str}_{cs_ratio_str}m_of_{num_points}ds_uniform.las')
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point Columbus Circle PC Using {basis_str}"
    metadata = f"{cs_percentage_str}% CS Ratio | {num_points} points (Uniform Downsampled) | NYCOpenData Columbus Circle 987217 Segmented1 PC | {measurement_type_str} Measurement | {basis_str} | CVXPY ECOS (Embedded Conic Solver) Reconstruction"
    
    # ---------------------
    
    # Import .las file
    nycpc.setLasCoords(inputpath)
    # Show Open3D plot of point cloud
    nycpc.showPC()
    # Multilevel 1D DWT
    cx, cy, cz = nycpc.transformDFT1D() # Return flattened and unflattened DWT coefficients of each of the 3 dimensions
    # IF NEEDED: Perform thresholding
        # fcx = CSPCdwt.applyThresholding(fcx, threshold=1e-14)
        # fcy = CSPCdwt.applyThresholding(fcy, threshold=1e-14)
        # fcz = CSPCdwt.applyThresholding(fcz, threshold=1e-14)
    # Create measurement matrix
    n_coeffs = len(cx)
    m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
    phi = CSPCdft.generateMeasurementMatrix(m, n_coeffs, type=measurement_type)
    # Measure (subsample) each of the point cloud dimensions (y = Cx)
    m_x, m_y, m_z = CSPCdft.measure1D(Phi = phi, x_coeffs=cx, y_coeffs=cy, z_coeffs=cz)
    # Reconstruct each point cloud dimension via L! Minimization and transform back from DWT
    solvertime = nycpc.reconstructCVXPY(m_x, m_y, m_z, phi)
    # IF NEEDED: Show Open3D plot of reconstructed point cloud
    nycpc.showReconstructedPC()
    # Export reconstructed point cloud as a .las file
    nycpc.writeReconstructedLas(path=outputpath)
    # Get reconstruction errors as a dictionary
    reconstructionErrors = nycpc.calculateReconstructionError()
    # Calculate sparsity of each basis-transformed coordinates
    s = CSPCdft.calculateSparsity(cx, cy, cz)
    # Export error and solver time to a .txt file
    CSPCdft.exportReconstructionInfo(outputfile=outputpath, info=metadata, errors=reconstructionErrors, solve_time=solvertime, sparsity=s)
    # Plot and export plots of both original point cloud and reconstructed point cloud
    nycpc.plotPCs(main_title=plot_title, e_angle=35, a_angle=-215,outputfile=outputpath, fileformat='pdf')
    # IF NEEDED: Export coordinates of original and reconstructed point clouds as .csv or .npy
    nycpc.exportCoords(outputpath, 'npy')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    
    
    
    # Example of creating my own point cloud with the PC's reconstructed coordinate instance fields
    # and then reconstructing it with the DWT and the IDWT
    '''
    pc1 = CSPointCloud()
    n_points = 25000
    pc1.setManualCoords(10*np.exp(np.linspace(0, 5, n_points))+1000*np.random.rand(n_points), 2500*np.sin(np.linspace(1, 5000*np.pi, n_points))+250*np.random.rand(n_points), np.linspace(1, 1000, n_points))
    pc1.showPC()
    #pc1.writeLas(path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/output.las')
    fcx, cx, fcy, cy, fcz, cz = pc1.transform_dwt1D()
    pc1.x_r = CSPointCloud.transform_idwt1D(fcx, cx)
    pc1.y_r = CSPointCloud.transform_idwt1D(fcy, cy)
    pc1.z_r = CSPointCloud.transform_idwt1D(fcz, cz)
    pc1.showReconstructedPC()
    '''
          
    """ This code performs 1D CS """
    '''
    t = np.linspace(0, np.pi, num=1000)
    test = 10*np.random.rand(1000, 1)
    x = np.exp(t) * np.sin(2*np.pi*t) + np.exp(2*np.random.rand(1000))
    data_a = CS1D(x, t)
    data_a.plot1D()
    cA, cD = pywt.dwt(x, 'haar')
    x_recon = pywt.idwt(cA = None, cD=cD, wavelet='haar')
    data_a_recon = CS1D(x_recon, t)
    data_a_recon.plot1D()
    '''    
    
    
    """ This code below attempts PC 3D compressed sensing with measuring done before transformation """
    '''
   pc_test = CSPointCloud_new()
   pc_test.setLasCoords('D:/Documents/Thesis_CS/Point_Cloud_Inputs/helix256.las')
   #n_points = 25000
   #pc_test.setManualCoords(10*np.exp(np.linspace(0, 5, n_points))+1000*np.random.rand(n_points), 2500*np.sin(np.linspace(1, 5000*np.pi, n_points))+250*np.random.rand(n_points), np.linspace(1, 1000, n_points))
   #pc_test.showPC()
   cs_ratio = 0.5
   n_coeffs = len(pc_test.x)
   m = int(n_coeffs * cs_ratio)  # Use 50% measurements for compression
   phi = CSPointCloud_new.generateMeasurementMatrix(m, n_coeffs)
   pc_test.measure1D(phi)
   CSPointCloud_new.showPC_DEBUG(pc_test.x_m, pc_test.y_m, pc_test.z_m, name='Measured Point Cloud')
   #pc_test.writeMeasuredLas()
   fcx, cx, fcy, cy, fcz, cz = pc_test.transform_dwt1D()
   
   tt, xx = pc_test.reconstructCVXPY(measured_x=fcx, coeffs_template_x=cx, measured_y=fcy, coeffs_template_y=cy, measured_z=fcy, coeffs_template_z=cy, phi=phi)
   pc_test.showReconstructedPC()
   pc_test.writeReconstructedLas()
   
   x_reconnn, time_reconnn = CSPointCloud_new.OLD_reconstructCVXPY(fcx, phi, cx)
   '''
    
                         