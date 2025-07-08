import numpy as np
import laspy
import open3d as o3d
import scipy.fftpack as spfft
import pywt
import cvxpy as cp
from cosamp import cosamp
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os
import ray
import point_cloud_utils as pcu
#import CSPCconfig # Project-specific config variables

""" ------------------------- Global Variables --------------------------"""
from CSPCconfig import INPUT_PATH_PCLAS, OUTPUT_PATH_PCLAS, OUTPUT_PATH_PLOTS
from CSPCconfig import OUTPUT_PATH_COMPILED

# Custom colormap
c_dict = {
    'red':   [(0.0, 0.0, 0.0), (0.33, 0.0, 0.0), (0.66, 1.0, 1.0), (1.0, 1.0, 1.0)],
    'green': [(0.0, 0.0, 0.0), (0.33, 1.0, 1.0), (0.66, 1.0, 1.0), (1.0, 0.0, 0.0)],
    'blue':  [(0.0, 1.0, 1.0), (0.33, 1.0, 1.0), (0.66, 0.0, 0.0), (1.0, 0.0, 0.0)]
}



class CSPCdwt:
    """ --------------- The cspc (CSPCdwt) library class. --------------- """
    ''' 
    A CSPCdwt object consists of the following attributes/instance fields:
        - x     (x coordinate)
        - y     (y coordinate)
        - z     (z coordinate)
        - x_r   (reconstructed x coordinate)
        - y_r   (reconstructed y coordinate)
        - z_r   (reconstructed z coordinate)
        - k     (sparsity of basis-transformed point cloud)
    '''
    
    # def __init__(self):
            
    def setLasCoords(self, pcpath, scaled = True):
        """ This method extracts the x,y,z coords from a .las file and
        fils up a CSPCdwt object with these coords. """
        """ By default, this method extracts the scaled values because scaled values
        are better for compressed sensing. """
        
        ''' TO DO ___________________'''
        ''' Convert the self.x,.y,.z instance fields to np arrays rather than
        incompatible laspy.point.dims.ScaledArrayView type'''
        las = laspy.read(os.path.normpath(pcpath))
        # Scaled values
        if(scaled):
            self.x = np.array(las.x)
            self.y = np.array(las.y)
            self.z = np.array(las.z)
        # Raw values
        else:
            self.x = np.array(las.X)
            self.y = np.array(las.Y)
            self.z = np.array(las.Z)
         
    def setManualCoords(self, x, y, z):  
        """This method accepts 1D numpy arrays for each of the x,y,z coords
        and fills up a CSPCdwt object with these coords. """ 
        self.x = x
        self.y = y
        self.z = z
       
    def showPC(self, name='Point Cloud'):
        """ This method plots the CSPCdwt object with Open3D
        as a separate window. """
        # Take each coordinate and arrange each dimension by columns:
        point_data = np.stack([self.x, self.y, self.z], axis=0).transpose((1, 0)) 
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        o3d.visualization.draw_geometries([geom], window_name=name) # Show point cloud
        # Note: Hit "r" to reset PC view in Open3D
    
    def showReconstructedPC(self, name='Reconstructed Point Cloud'):
        """ This method plots the CSPCdwt object with Open3D
        as a separate window from its reconstructed coordinates. """  
        # Take each coordinate and arrange each dimension by columns:
        point_data = np.stack([self.x_r, self.y_r, self.z_r], axis=0).transpose((1, 0)) 
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        o3d.visualization.draw_geometries([geom], window_name=name) # Show point cloud
        # Note: Hit "r" to reset PC view in Open3D
        return 0
    
    def writeLas(self, path = os.path.join(INPUT_PATH_PCLAS, 'originalpc.las'), las_point_format = 3, las_version = "1.2"):
        """ "This method stores the CSPCdwt object's coordinates' as a .las file. """
        header = laspy.LasHeader(version=las_version, point_format=las_point_format)
        las = laspy.LasData(header)
        las.x = self.x
        las.y = self.y
        las.z = self.z
        las.write(os.path.normpath(path))
        
    def writeReconstructedLas(self, path = os.path.join(OUTPUT_PATH_PCLAS, 'reconstructedpc.las'), las_point_format = 3, las_version = "1.2"):
        """ "This method stores the CSPCdwt object's reconstructed coordinates' as a .las file. """
        header = laspy.LasHeader(version=las_version, point_format=las_point_format)
        las = laspy.LasData(header)
        las.x = self.x_r
        las.y = self.y_r
        las.z = self.z_r
        las.write(os.path.normpath(path))       
    
    def performDownsample(self, dsrate, showPC=True, dstype='uniform'):
        """ This method performs uniform downsampling from Open3D 
            on the CSPCdwt object. """
        """
        - Accepts x,y,z coordinate 1D ndarrays
        - Returns coordinates as (m x 3) matrix and shape
        - k is the downsampling rate (i.e. if k=2, 1/2 of the samples are kept).
        """
        # An open3D object has to be created in order to downsample
        point_data = np.stack([self.x, self.y, self.z], axis=0).transpose((1, 0)) # Take each coordinate in .las file and arrange it in an nd array
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        
        # Downsampling function
        if(dstype=='uniform'):
            down_geom = geom.uniform_down_sample(dsrate)
        elif(dstype=='voxel'):
            down_geom = geom.voxel_down_sample(voxel_size=0.05)
        else:
            raise(ValueError("Invalid downsampling type. Select \"uniform\" or \"voxel\"."))
        
        ''' WHEN VOXEL DOWNSAMPLING IS DONE, THE OUTPUT IS LIST OF MATRICES AND NOT 1 MATRIX '''
        
        # Show point cloud
        if(showPC):
            o3d.visualization.draw_geometries([down_geom], window_name='Downsampled Point Cloud')
            
        np_points = np.asarray(down_geom.points) # Returns (m x 3) ndarray of x,y,z coordinates
        #dsshape = np_points[1]
        
        # Trim away the last row, if total amount of downsampled points is not an even value
            # Note: This is because the DWT only works on even-valued amount of points. Otherwise, the 
            # points get padded and the original and reconstructed point clouds are not the same shape. 
        if(np_points.shape[0] % 2 != 0):
            print("Odd amount of points. Trimming....")
            np_points_trim = np.delete(np_points, np_points.shape[0]-1, axis=0)
            self.x = np_points_trim[:, 0] 
            self.y = np_points_trim[:, 1]
            self.z = np_points_trim[:, 2]
        else:
            self.x = np_points[:, 0] 
            self.y = np_points[:, 1]
            self.z = np_points[:, 2]
            

        #return dscoords, np_points
        #return np_points, np_points.shape # Returns matrix where each columns is downsampled coordinate for each dimension
    
    def transformDWT1D(self, wavelet, padding=False):
        """ This method applies 1D DWT and returns flattened coefficients with padding if needed
         for each dimension of the CSPCdwt object. 
         This method also returns the unflattened
         Wavelet coefficients to use a template for unflattening and restoring the shape of the 
         list of ndarray coefficients. """
        coeffs_x = pywt.wavedec(self.x, wavelet)
        flat_coeffs_x = np.hstack(coeffs_x)
        coeffs_y = pywt.wavedec(self.y, wavelet)
        flat_coeffs_y = np.hstack(coeffs_y)
        coeffs_z = pywt.wavedec(self.z, wavelet)
        flat_coeffs_z = np.hstack(coeffs_z)
        
        if(padding):
            # Ensure consistent size by padding to the next even size
            padded_length_x = (len(flat_coeffs_x) + 1) if len(flat_coeffs_x) % 2 != 0 else len(flat_coeffs_x)
            flat_coeffs_x = np.pad(flat_coeffs_x, (0, padded_length_x - len(flat_coeffs_x)))
            # Ensure consistent size by padding to the next even size
            padded_length_y = (len(flat_coeffs_y) + 1) if len(flat_coeffs_y) % 2 != 0 else len(flat_coeffs_y)
            flat_coeffs_y = np.pad(flat_coeffs_y, (0, padded_length_y - len(flat_coeffs_y)))
            # Ensure consistent size by padding to the next even size
            padded_length_z = (len(flat_coeffs_z) + 1) if len(flat_coeffs_z) % 2 != 0 else len(flat_coeffs_z)
            flat_coeffs_z = np.pad(flat_coeffs_z, (0, padded_length_z - len(flat_coeffs_z)))
        
        return flat_coeffs_x, coeffs_x, flat_coeffs_y, coeffs_y, flat_coeffs_z, coeffs_z
 
    def applyThresholding(arr, threshold):
        """" This method applies thresholding to the sparse signal before measurement to make even sparser. """
        """ Or this is done in place of the measurement matrix once the signal is in the sparse domain. """
        # calculateSparsity()
        # perform thresholding here to improve sparsity
        
        # For elements that are greater than or equal to the threshold, keep them the same
        # For elements that are < the threshold, replace them with zeros
        thresholded_arr = np.where(arr >= threshold, arr, 0)
        return thresholded_arr
    
    @staticmethod
    def transformIDWT1D(flat_coeffs, unflat_coeffs, wavelet):
        """ This method accepts the flattened coeffs of a 1D signal/dimension, 
        unflattens the coeffs, and reconstructs based on the DWT coefficients. """
        # flat_coeffs: the flattened ndarray of the multi-level discrete Wavelet decomposition coefficients for one of the PC's dimensions.
        # wavelet: the string of the chosen Wavelet to use for the DWT
        # unflat_coeffs: the raw unflattened (list of ndarrays) result of multilevel discrete Wavelet decomposition (wavedec).
        #                that is used as a template for unflattening and restoring the shape of the Wavelet coefficients.
        coeffs = []
        idx = 0
        for c in unflat_coeffs: # unflat_coeffs is the list of ndarrays used as template to unflatten the flat_coeffs
            size = len(c) # size of each ndarray
            coeffs.append(flat_coeffs[idx:idx + size])
            idx += size
        return pywt.waverec(coeffs, wavelet)
    
    def reconstructCVXPY(self, measured_x, coeffs_template_x, measured_y, coeffs_template_y, measured_z, coeffs_template_z, phi, wavelet):
        """ This method performs L1 minimization using the CVXPY library. """
        """ In order for each of the 3 dimensions to be reconstructed
        independently, the problem is solved 3 times with 3 different parameters."""
        """ Returns the reconstructed (x,y & z) coordinates along with time
        taken for reconstruction algorithm."""
        
        """ TO DO: ______________"""
        """ Make this function general purpose for any kind of basis by
        adding an instance field for the CSPCdwt object named 'basis' 
        and based on the 'basis" instance field, have this function do different
        things according to the basis. For example, since DWT requires flattening/unflattening
        of the coefficients, have this call a flattening/unflattening function beforehand
        to remove the 'coeffs_templates' parameters. Instead add 3 other
        instance fields to the CSPCdwt object that stores the measured
        basis coefficients. For DWT, I can have them store that list of ndarrays"""
        
        '''
        if(self.basis=='DWT'):
            flatten()
            
        
        '''
        
        n = phi.shape[1] # Amount of coefficients
        
        # Define the optimization variable (the recovered DWT coefficients)
        s = cp.Variable(n)
        # Reusable parameters
        if(measured_x.size == measured_y.size == measured_z.size):
            y = cp.Parameter(measured_x.size)    
        else:
            raise ValueError("Point Cloud dimensions are mismatched in length. Unable to define \"y\".")

        # Define the L1-norm objective and constraint (y = Phi * x) and problem
        objective = cp.Minimize(cp.norm1(s))  # L1 minimization
        constraints = [phi @ s == y]
        prob = cp.Problem(objective, constraints)

        start_time = time()
        
        # Solve the optimization problems for x coordinates
        y.value = measured_x
        prob.solve(verbose = True)
        x_sparse_r = s.value # Sparse reconstructed value
        
        # Solve the optimization problems for y coordinates
        y.value = measured_y
        prob.solve(verbose = True)
        y_sparse_r = s.value # Sparse reconstructed value
        
        # Solve the optimization problems for z coordinates
        y.value = measured_z
        prob.solve(verbose = True)
        z_sparse_r = s.value # Sparse reconstructed value
        
        # Time of optimization algorithm
        end_time = time()
        solve_time = end_time - start_time

        if prob.status != cp.OPTIMAL:
            raise ValueError(f"Optimization failed: {prob.status}")

        # Inverse transform back to original domain
        self.x_r = CSPCdwt.transformIDWT1D(x_sparse_r, coeffs_template_x, wavelet) #TODO
        self.y_r = CSPCdwt.transformIDWT1D(y_sparse_r, coeffs_template_y, wavelet) #TODO
        self.z_r = CSPCdwt.transformIDWT1D(z_sparse_r, coeffs_template_z, wavelet) #TODO
        
        print(f"Solver Time: {solve_time} [s]")
        return solve_time

    def reconstructCVXPY_ray(self, measured_x, coeffs_template_x, measured_y, coeffs_template_y, measured_z, coeffs_template_z, phi, wavelet, n_cpus=2):
            """ This method performs L1 minimization using the CVXPY library. """
            """ In order for each of the 3 dimensions to be reconstructed
            independently, the problem is solved 3 times with 3 different parameters."""
            """ Returns the reconstructed (x,y & z) coordinates along with time
            taken for reconstruction algorithm."""
    
            ray.init(ignore_reinit_error=True)
    
            start_time = time()
            
            @ray.remote(num_cpus=n_cpus)
            def reconstruct(measured_coord, phi):
                # Amount of coefficients to reconstruct:
                n = phi.shape[1]
                # Define the optimization variable (the recovered DWT coefficients:
                s = cp.Variable(n)
                # Reusable parameter:
                y = cp.Parameter(len(measured_coord))
                y.value = measured_coord
                # Define the L1-norm objective and constraint (y = Phi * x) and problem
                objective = cp.Minimize(cp.norm1(s))
                constraints = [phi @ s == y]
                prob = cp.Problem(objective, constraints)
                prob.solve(verbose=True)
                return s.value # Return sparse reconstructed value
            
            # NOTE: Using Ray for large objects, like large arrays can cause performance issues!
            # So, I am using ray.put() to store the large object in the Ray object store.
            measured_x_ref = ray.put(measured_x)
            measured_y_ref = ray.put(measured_y)
            measured_z_ref = ray.put(measured_z)
            phi_ref = ray.put(phi)
            
            # Parallel processing reconstruction:
            future_x = reconstruct.remote(measured_x_ref, phi_ref)
            future_y = reconstruct.remote(measured_y_ref, phi_ref)
            future_z = reconstruct.remote(measured_z_ref, phi_ref)
            result_x_sparse_r = ray.get(future_x)
            result_y_sparse_r = ray.get(future_y)
            result_z_sparse_r = ray.get(future_z)
            
            # Time of optimization algorithm
            end_time = time()
            solve_time = end_time - start_time
            
            # Inverse transform back to original domain
            self.x_r = CSPCdwt.transformIDWT1D(result_x_sparse_r, coeffs_template_x, wavelet) #TODO
            self.y_r = CSPCdwt.transformIDWT1D(result_y_sparse_r, coeffs_template_y, wavelet) #TODO
            self.z_r = CSPCdwt.transformIDWT1D(result_z_sparse_r, coeffs_template_z, wavelet) #TODO
            
            print(f"Solver Time: {solve_time} [s]")
            
            ray.shutdown()
            print("ray.shutdown() called.")
            
            return solve_time

    def reconstructCosamp(self, measured_x, coeffs_template_x, measured_y, coeffs_template_y, measured_z, coeffs_template_z, phi, wavelet, sx, sy, sz, tol=1e-10, max_iter=1000, norm='ortho'):
        """ This method performs L1 minimization using the CoSaMP algorithm. """
        """ In order for each of the 3 dimensions to be reconstructed
        independently, the problem is solved 3 times with 3 different parameters."""
        """ Returns the reconstructed (x,y & z) coordinates along with time
        taken for reconstruction algorithm."""
        
        if(measured_x.size != measured_y.size != measured_z.size):
            raise ValueError("Point Cloud dimensions are mismatched in length. Unable to define \"y\".")
        start_time = time()
        
        # Solve the optimization problems for x coordinates
        y_x = cosamp.cosamp(phi, measured_x, s=sx, tol=tol, max_iter=max_iter) # CS via matching pursuit (s = known maximum number of non-zero elements in the signal being reconstructed)
        x_sparse_r = y_x # Sparse reconstructed value
        
        # Solve the optimization problems for y coordinates
        y_y = cosamp.cosamp(phi, measured_y, s=sy, tol=tol, max_iter=max_iter) # CS via matching pursuit
        y_sparse_r = y_y # Sparse reconstructed value
        
        # Solve the optimization problems for z coordinates
        y_z = cosamp.cosamp(phi, measured_z, s=sz, tol=tol, max_iter=max_iter) # CS via matching pursuit (#epsilon=1.e-10, max_iter=10)
        z_sparse_r = y_z # Sparse reconstructed value
        
        # Time of optimization algorithm
        end_time = time()
        solve_time = end_time - start_time

        # Inverse transform back to original domain
        self.x_r = CSPCdwt.transformIDWT1D(x_sparse_r, coeffs_template_x, wavelet) #TODO
        self.y_r = CSPCdwt.transformIDWT1D(y_sparse_r, coeffs_template_y, wavelet) #TODO
        self.z_r = CSPCdwt.transformIDWT1D(z_sparse_r, coeffs_template_z, wavelet) #TODO
        
        print(f"Solver Time: {solve_time} [s]")
        return solve_time, x_sparse_r, y_sparse_r, z_sparse_r
        
        pass    

    def calculateReconstructionError(self):
        """ This method calculates and outputs the 2-Norm Error, MSE Error, RMSE Error, and MAE error of the Point Cloud object 
        to the console and returns them as a dictionary. """
        
        point_cloud = np.column_stack((self.x, self.y, self.z))
        point_cloud_reconstructed = np.column_stack((self.x_r, self.y_r, self.z_r))
        
        l2error_val = np.linalg.norm(point_cloud - point_cloud_reconstructed) / np.linalg.norm(point_cloud)
        print("Reconstruction Errors ------------------------- \n")
        print(f"2-Norm Error: {l2error_val:.4f}")
        # Mean Squared Error
        mse_val = mean_squared_error(point_cloud, point_cloud_reconstructed)
        print("MSE: ", mse_val)
        # Root Mean Squared Error
        rmse_val = math.sqrt(mse_val)
        print("RMSE: ", rmse_val)
        # Mean Absolute Error
        mae_val = mean_absolute_error(point_cloud, point_cloud_reconstructed)
        print("MAE: ", mae_val)
        
        # Chamfer Distance
        cd_val = pcu.chamfer_distance(point_cloud, point_cloud_reconstructed)
        print("Chamfer Distance: ", cd_val)
        
        # Hausdorff Distance
        hd_val = pcu.hausdorff_distance(point_cloud, point_cloud_reconstructed)
        print("Hausdorff Distance: ", hd_val)
        
        # Earth Mover's Distance
        emd_val = 0 # TODO Disable EMD for now since it takes too long to calculate
        #emd_val, pi_val = pcu.earth_movers_distance(point_cloud, point_cloud_reconstructed) # Calculating EMD takes a VERY long time
        print("Earth Mover's Distance: ", emd_val)
        
        errors = dict(l2norm = l2error_val, MSE = mse_val, RMSE = rmse_val, MAE = mae_val, CD = cd_val, HD = hd_val, EMD = emd_val)
        
        return errors

    def plotPCs(self, main_title, e_angle=230, a_angle=-240, outputfile=os.path.join(OUTPUT_PATH_PCLAS, 'reconstructedpc.las'), fileformat='pdf', heightshown=True, pointsize=5, colormap='cool'):
        """ Plots the original and reconstructed point clouds and saves the plot as .pdf, .jpg, .png, or .svg. """
        """ Expects the output file to be the original output .las file path. """
        
        # Changes to .pdf file (or other specified formats: .png, .jpg, .svg) from .las
        outputpath = os.path.normpath(outputfile[:-3] + fileformat)
        
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(main_title)
        
        if(colormap == 'BGYR'):
            blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
            colormap = blue_green_yellow_red
        
        if(heightshown == True): # With elevation colors
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.x, self.y, self.z, c=self.z, cmap=colormap, s=pointsize)
            ax1.set_title("Original Point Cloud")
            ax1.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax1.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax1.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax1.set_xlabel('$X$', fontsize=15)
            ax1.set_ylabel('$Y$', fontsize=15)
            ax1.set_zlabel('$Z$', fontsize=15)
    
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(self.x_r, self.y_r, self.z_r, c=self.z, cmap=colormap, s=pointsize)
            ax2.set_title("Reconstructed Point Cloud")
            ax2.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax2.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax2.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax2.set_xlabel('$X$', fontsize=15)
            ax2.set_ylabel('$Y$', fontsize=15)
            ax2.set_zlabel('$Z$', fontsize=15)
        else: # Without elevation values
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.x, self.y, self.z, c='b' , s=pointsize)
            ax1.set_title("Original Point Cloud")
            ax1.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax1.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax1.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax1.set_xlabel('$X$', fontsize=15)
            ax1.set_ylabel('$Y$', fontsize=15)
            ax1.set_zlabel('$Z$', fontsize=15)
    
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(self.x_r, self.y_r, self.z_r, c='b', s=pointsize)
            ax2.set_title("Reconstructed Point Cloud")
            ax2.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax2.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax2.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax2.set_xlabel('$X$', fontsize=15)
            ax2.set_ylabel('$Y$', fontsize=15)
            ax2.set_zlabel('$Z$', fontsize=15)
        
        ax1.view_init(elev=e_angle, azim=a_angle)
        ax2.view_init(elev=e_angle, azim=a_angle)

        plt.savefig(outputpath)
        plt.show()

    def exportCoords(self, outputfileoriginal=os.path.join(INPUT_PATH_PCLAS, 'originalpc.las'), outputfilereconstructed=os.path.join(OUTPUT_PATH_PCLAS, 'reconstructedpc.las'), outputformat='csv', exportchoice='both'):
        """ This method saves and exports the coordinates of both the original point cloud and the reconstructed point cloud
        into .csv or .npy file formats. """
        
        point_cloud = np.column_stack((self.x, self.y, self.z))
        point_cloud_reconstructed = np.column_stack((self.x_r, self.y_r, self.z_r))
        # Paths for exporting the npy file(s)
        outputoriginal = os.path.normpath(outputfileoriginal[:-3] + outputformat)
        outputreconstructed = os.path.normpath(outputfilereconstructed[:-3] + outputformat)
        
        if(exportchoice == 'both'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputoriginal, point_cloud, delimiter = ',')
                np.savetxt(outputreconstructed, point_cloud_reconstructed, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputoriginal, point_cloud)
                np.save(outputreconstructed, point_cloud_reconstructed)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')  
        elif(exportchoice == 'original'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputoriginal, point_cloud, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputoriginal, point_cloud)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')
        elif(exportchoice == 'reconstructed'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputreconstructed, point_cloud_reconstructed, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputreconstructed, point_cloud_reconstructed)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')
        else:
            raise ValueError('Incorrect export choice. Choose \'both\', \'original\', or \'reconstructed\'.')
        
    def plotPCDistribution(self, title='Point Cloud Distribution'):
        """ This method plots the x, y, and z coordinates of the point cloud by index to show the distribution shape of each dimension in the point cloud. """
        
        plt.scatter(np.arange(len(self.x[0:100])), self.x[0:100])
        plt.ylabel("X")
        plt.xlabel("Index")
        plt.title(title)
        plt.show()
        
        plt.scatter(np.arange(len(self.y[0:100])), self.y[0:100])
        plt.ylabel("Y")
        plt.xlabel("Index")
        plt.title(title)
        plt.show()
        
        plt.scatter(np.arange(len(self.z[0:100])), self.z[0:100])
        plt.ylabel("Z")
        plt.xlabel("Index")
        plt.title(title)
        plt.show()
    
class CSPCdct:
    """ --------------- The cspc (CSPCdct) library class. --------------- """
    ''' 
    A CSPCdwt object consists of the following attributes/instance fields:
        - x     (x coordinate)
        - y     (y coordinate)
        - z     (z coordinate)
        - x_r   (reconstructed x coordinate)
        - y_r   (reconstructed y coordinate)
        - z_r   (reconstructed z coordinate)
        - k     (sparsity of basis-transformed point cloud)
        - basis_type (basis used for sparse transformation)
        - x_m (measured x coordinate basis coefficients)
        - y_m (measured y coordinate basis coefficients)
        - z_m (measured z coordinate basis coefficients)
        ??? - measurement_type (matrix used for measurememt matrix)
        ??? - CS_ratio
    '''
    
    # def __init__(self):
            
    def setLasCoords(self, pcpath, scaled = True):
        """ This method extracts the x,y,z coords from a .las file and
        fils up a CSPCdct object with these coords. """
        """ By default, this method extracts the scaled values because scaled values
        are better for compressed sensing. """
        
        ''' TO DO ___________________'''
        ''' Convert the self.x,.y,.z instance fields to np arrays rather than
        incompatible laspy.point.dims.ScaledArrayView type'''
        las = laspy.read(os.path.normpath(pcpath))
        # Scaled values
        if(scaled):
            self.x = np.array(las.x)
            self.y = np.array(las.y)
            self.z = np.array(las.z)
        # Raw values
        else:
            self.x = np.array(las.X)
            self.y = np.array(las.Y)
            self.z = np.array(las.Z)
         
    def setManualCoords(self, x, y, z):  
        """This method accepts 1D numpy arrays for each of the x,y,z coords
        and fills up a CSPCdct object with these coords. """ 
        self.x = x
        self.y = y
        self.z = z
       
    def showPC(self, name='Point Cloud'):
        """ This method plots the CSPCdct object with Open3D
        as a separate window. """
        # Take each coordinate and arrange each dimension by columns:
        point_data = np.stack([self.x, self.y, self.z], axis=0).transpose((1, 0)) 
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        o3d.visualization.draw_geometries([geom], window_name=name) # Show point cloud
        # Note: Hit "r" to reset PC view in Open3D
    
    def showReconstructedPC(self, name='Reconstructed Point Cloud'):
        """ This method plots the CSPCdct object with Open3D
        as a separate window from its reconstructed coordinates. """  
        # Take each coordinate and arrange each dimension by columns:
        point_data = np.stack([self.x_r, self.y_r, self.z_r], axis=0).transpose((1, 0)) 
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        o3d.visualization.draw_geometries([geom], window_name=name) # Show point cloud
        # Note: Hit "r" to reset PC view in Open3D
        return 0
    
    def writeLas(self, path = os.path.join(INPUT_PATH_PCLAS, 'originalpc.las'), las_point_format = 3, las_version = "1.2"):
        """ "This method stores the CSPCdwt object's coordinates' as a .las file. """
        header = laspy.LasHeader(version=las_version, point_format=las_point_format)
        las = laspy.LasData(header)
        las.x = self.x
        las.y = self.y
        las.z = self.z
        las.write(os.path.normpath(path))
        
    def writeReconstructedLas(self, path = os.path.join(OUTPUT_PATH_PCLAS, 'reconstructedpc.las'), las_point_format = 3, las_version = "1.2"):
        """ "This method stores the CSPCdwt object's reconstructed coordinates' as a .las file. """
        header = laspy.LasHeader(version=las_version, point_format=las_point_format)
        las = laspy.LasData(header)
        las.x = self.x_r
        las.y = self.y_r
        las.z = self.z_r
        las.write(os.path.normpath(path))       
    
    def performDownsample(self, dsrate, showPC=True, dstype='uniform'):
        """ This method performs uniform downsampling from Open3D 
            on the CSPCdct object. """
        """
        - Accepts x,y,z coordinate 1D ndarrays
        - Returns coordinates as (m x 3) matrix and shape
        - k is the downsampling rate (i.e. if k=2, 1/2 of the samples are kept).
        """
        # An open3D object has to be created in order to downsample
        point_data = np.stack([self.x, self.y, self.z], axis=0).transpose((1, 0)) # Take each coordinate in .las file and arrange it in an nd array
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        
        # Downsampling function
        if(dstype=='uniform'):
            down_geom = geom.uniform_down_sample(dsrate)
        elif(dstype=='voxel'):
            down_geom = geom.voxel_down_sample(voxel_size=0.05)
        else:
            raise(ValueError("Invalid downsampling type. Select \"uniform\" or \"voxel\"."))
        
        ''' WHEN VOXEL DOWNSAMPLING IS DONE, THE OUTPUT IS LIST OF MATRICES AND NOT 1 MATRIX '''
        
        # Show point cloud
        if(showPC):
            o3d.visualization.draw_geometries([down_geom], window_name='Downsampled Point Cloud')
            
        np_points = np.asarray(down_geom.points) # Returns tuple of coordinate matrix and shape
        #dsshape = np_points[1]
        self.x = np_points[:,0] 
        self.y = np_points[:,1]
        self.z = np_points[:,2]
        #return dscoords, np_points
        #return np_points, np_points.shape # Returns matrix where each columns is downsampled coordinate for each dimension
    
    def transformDCT1D(self, norm='ortho', padding=False):
        """ This method applies 1D DCT and returns DCT coefficients with padding if needed,
         for each dimension of the CSPCdct object. """
        coeffs_x = spfft.dct(self.x, norm=norm)
        coeffs_y = spfft.dct(self.y, norm=norm)
        coeffs_z = spfft.dct(self.z, norm=norm)
        
        """
        if(padding):
            # Ensure consistent size by padding to the next even size
            padded_length_x = (len(flat_coeffs_x) + 1) if len(flat_coeffs_x) % 2 != 0 else len(flat_coeffs_x)
            flat_coeffs_x = np.pad(flat_coeffs_x, (0, padded_length_x - len(flat_coeffs_x)))
            # Ensure consistent size by padding to the next even size
            padded_length_y = (len(flat_coeffs_y) + 1) if len(flat_coeffs_y) % 2 != 0 else len(flat_coeffs_y)
            flat_coeffs_y = np.pad(flat_coeffs_y, (0, padded_length_y - len(flat_coeffs_y)))
            # Ensure consistent size by padding to the next even size
            padded_length_z = (len(flat_coeffs_z) + 1) if len(flat_coeffs_z) % 2 != 0 else len(flat_coeffs_z)
            flat_coeffs_z = np.pad(flat_coeffs_z, (0, padded_length_z - len(flat_coeffs_z)))
        """  
        return coeffs_x, coeffs_y, coeffs_z

    def applyThresholding(arr, threshold):
        """" This method applies thresholding to the sparse signal before measurement to make even sparser. """
        """ Or this is done in place of the measurement matrix once the signal is in the sparse domain. """
        # calculateSparsity()
        # perform thresholding here to improve sparsity
        
        # For elements that are greater than or equal to the threshold, keep them the same
        # For elements that are < the threshold, replace them with zeros
        thresholded_arr = np.where(arr >= threshold, arr, 0)
        return thresholded_arr

    @staticmethod
    def transformIDCT1D(coeffs, norm='ortho'):
        """ This method accepts the DCT coeffs of a 1D signal/dimension, 
        and reconstructs based on the DCT coefficients. """
        # flat_coeffs: the flattened ndarray of the multi-level discrete Cosine Transformation coefficients for one of the PC's dimensions.
        # wavelet: the string of the chosen Wavelet to use for the DWC

        return spfft.idct(coeffs, norm=norm)
    
    def reconstructCVXPY(self, measured_x, measured_y, measured_z, phi, norm='ortho'):
        """ This method performs L1 minimization using the CVXPY library. """
        """ In order for each of the 3 dimensions to be reconstructed
        independently, the problem is solved 3 times with 3 different parameters."""
        """ Returns the reconstructed (x,y & z) coordinates along with time
        taken for reconstruction algorithm."""
        
        n = phi.shape[1] # Amount of coefficients
        
        # Define the optimization variable (the recovered DWT coefficients)
        s = cp.Variable(n)
        # Reusable parameters
        if(measured_x.size == measured_y.size == measured_z.size):
            y = cp.Parameter(measured_x.size)  
        else:
            raise ValueError("Point Cloud dimensions are mismatched in length. Unable to define \"y\".")

        # Define the L1-norm objective and constraint (y = Phi * x) and problem
        objective = cp.Minimize(cp.norm1(s))  # L1 minimization
        constraints = [phi @ s == y]
        prob = cp.Problem(objective, constraints)

        start_time = time()
        
        # Solve the optimization problems for x coordinates
        y.value = measured_x
        prob.solve(verbose = True)
        x_sparse_r = s.value # Sparse reconstructed value
        
        # Solve the optimization problems for y coordinates
        y.value = measured_y
        prob.solve(verbose = True)
        y_sparse_r = s.value # Sparse reconstructed value
        
        # Solve the optimization problems for z coordinates
        y.value = measured_z
        prob.solve(verbose = True)
        z_sparse_r = s.value # Sparse reconstructed value
        
        # Time of optimization algorithm
        end_time = time()
        solve_time = end_time - start_time

        if prob.status != cp.OPTIMAL:
            raise ValueError(f"Optimization failed: {prob.status}")

        # Inverse transform back to original domain
        self.x_r = CSPCdct.transformIDCT1D(x_sparse_r, norm=norm) #TODO
        self.y_r = CSPCdct.transformIDCT1D(y_sparse_r, norm=norm) #TODO
        self.z_r = CSPCdct.transformIDCT1D(z_sparse_r, norm=norm) #TODO
        
        print(f"Solver Time: {solve_time} [s]")
        return solve_time

    def reconstructCVXPY_ray(self, measured_x, measured_y, measured_z, phi, norm='ortho', n_cpus=2):
            """ This method performs L1 minimization using the CVXPY library. """
            """ In order for each of the 3 dimensions to be reconstructed
            independently, the problem is solved 3 times with 3 different parameters."""
            """ Returns the reconstructed (x,y & z) coordinates along with time
            taken for reconstruction algorithm."""
    
            ray.init(ignore_reinit_error=True)
    
            start_time = time()
            
            @ray.remote(num_cpus=n_cpus)
            def reconstruct(measured_coord, phi):
                # Amount of coefficients to reconstruct:
                n = phi.shape[1]
                # Define the optimization variable (the recovered DWT coefficients:
                s = cp.Variable(n)
                # Reusable parameter:
                y = cp.Parameter(len(measured_coord))
                y.value = measured_coord
                # Define the L1-norm objective and constraint (y = Phi * x) and problem
                objective = cp.Minimize(cp.norm1(s))
                constraints = [phi @ s == y]
                prob = cp.Problem(objective, constraints)
                prob.solve(verbose=True)
                return s.value # Return sparse reconstructed value
            
            # NOTE: Using Ray for large objects, like large arrays can cause performance issues!
            # So, I am using ray.put() to store the large object in the Ray object store.
            measured_x_ref = ray.put(measured_x)
            measured_y_ref = ray.put(measured_y)
            measured_z_ref = ray.put(measured_z)
            phi_ref = ray.put(phi)
            
            # Parallel processing reconstruction:
            future_x = reconstruct.remote(measured_x_ref, phi_ref)
            future_y = reconstruct.remote(measured_y_ref, phi_ref)
            future_z = reconstruct.remote(measured_z_ref, phi_ref)
            result_x_sparse_r = ray.get(future_x)
            result_y_sparse_r = ray.get(future_y)
            result_z_sparse_r = ray.get(future_z)
            
            # Time of optimization algorithm
            end_time = time()
            solve_time = end_time - start_time
            
            # Inverse transform back to original domain
            self.x_r = CSPCdct.transformIDCT1D(result_x_sparse_r, norm=norm) #TODO
            self.y_r = CSPCdct.transformIDCT1D(result_y_sparse_r, norm=norm) #TODO
            self.z_r = CSPCdct.transformIDCT1D(result_z_sparse_r, norm=norm) #TODO
            
            
            print(f"Solver Time: {solve_time} [s]")
            
            ray.shutdown()
            print("ray.shutdown() called.")
            
            return solve_time

    def reconstructCosamp(): #TODO
        pass

    def calculateReconstructionError(self):
        """ This method calculates and outputs the 2-Norm Error, MSE Error, RMSE Error, and MAE error of the Point Cloud object 
        to the console and returns them as a dictionary. """
        
        point_cloud = np.column_stack((self.x, self.y, self.z))
        point_cloud_reconstructed = np.column_stack((self.x_r, self.y_r, self.z_r))
        
        l2error_val = np.linalg.norm(point_cloud - point_cloud_reconstructed) / np.linalg.norm(point_cloud)
        print("Reconstruction Errors ------------------------- \n")
        print(f"2-Norm Error: {l2error_val:.4f}")
        # Mean Squared Error
        mse_val = mean_squared_error(point_cloud, point_cloud_reconstructed)
        print("MSE: ", mse_val)
        # Root Mean Squared Error
        rmse_val = math.sqrt(mse_val)
        print("RMSE: ", rmse_val)
        # Mean Absolute Error
        mae_val = mean_absolute_error(point_cloud, point_cloud_reconstructed)
        print("MAE: ", mae_val)
        
        # Chamfer Distance
        cd_val = pcu.chamfer_distance(point_cloud, point_cloud_reconstructed)
        print("Chamfer Distance: ", cd_val)
        
        # Hausdorff Distance
        hd_val = pcu.hausdorff_distance(point_cloud, point_cloud_reconstructed)
        print("Hausdorff Distance: ", hd_val)
        
        # Earth Mover's Distance
        emd_val = 0 # TODO Disable EMD for now since it takes too long to calculate
        #emd_val, pi_val = pcu.earth_movers_distance(point_cloud, point_cloud_reconstructed) # Calculating EMD takes a VERY long time
        print("Earth Mover's Distance: ", emd_val)
        
        errors = dict(l2norm = l2error_val, MSE = mse_val, RMSE = rmse_val, MAE = mae_val, CD = cd_val, HD = hd_val, EMD = emd_val)
        
        return errors

    def plotPCs(self, main_title, e_angle=230, a_angle=-240, outputfile=os.path.join(OUTPUT_PATH_PCLAS, 'reconstructedpc.las'), fileformat='pdf', heightshown=True, pointsize=5, colormap='cool'):
        """ Plots the original and reconstructed point clouds and saves the plot as .pdf, .jpg, .png, or .svg. """
        """ Expects the output file to be the original output .las file path. """
        
        # Changes to .pdf file (or other specified formats: .png, .jpg, .svg) from .las
        outputpath = os.path.normpath(outputfile[:-3] + fileformat)
        
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(main_title)
        
        if(colormap == 'BGYR'):
            blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
            colormap = blue_green_yellow_red
        
        if(heightshown == True): # With elevation colors
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.x, self.y, self.z, c=self.z, cmap=colormap, s=pointsize)
            ax1.set_title("Original Point Cloud")
            ax1.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax1.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax1.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax1.set_xlabel('$X$', fontsize=15)
            ax1.set_ylabel('$Y$', fontsize=15)
            ax1.set_zlabel('$Z$', fontsize=15)
    
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(self.x_r, self.y_r, self.z_r, c=self.z, cmap=colormap, s=pointsize)
            ax2.set_title("Reconstructed Point Cloud")
            ax2.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax2.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax2.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax2.set_xlabel('$X$', fontsize=15)
            ax2.set_ylabel('$Y$', fontsize=15)
            ax2.set_zlabel('$Z$', fontsize=15)
        else: # Without elevation values
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.x, self.y, self.z, c='b' , s=pointsize)
            ax1.set_title("Original Point Cloud")
            ax1.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax1.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax1.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax1.set_xlabel('$X$', fontsize=15)
            ax1.set_ylabel('$Y$', fontsize=15)
            ax1.set_zlabel('$Z$', fontsize=15)
    
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(self.x_r, self.y_r, self.z_r, c='b', s=pointsize)
            ax2.set_title("Reconstructed Point Cloud")
            ax2.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax2.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax2.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax2.set_xlabel('$X$', fontsize=15)
            ax2.set_ylabel('$Y$', fontsize=15)
            ax2.set_zlabel('$Z$', fontsize=15)
        
        ax1.view_init(elev=e_angle, azim=a_angle)
        ax2.view_init(elev=e_angle, azim=a_angle)

        plt.savefig(outputpath)
        plt.show()

    def exportCoords(self, outputfileoriginal=os.path.join(INPUT_PATH_PCLAS, 'originalpc.las'), outputfilereconstructed=os.path.join(OUTPUT_PATH_PCLAS, 'reconstructedpc.las'), outputformat='csv', exportchoice='both'):
        """ This method saves and exports the coordinates of both the original point cloud and the reconstructed point cloud
        into .csv or .npy file formats. """
        
        point_cloud = np.column_stack((self.x, self.y, self.z))
        point_cloud_reconstructed = np.column_stack((self.x_r, self.y_r, self.z_r))
        # Paths for exporting the npy file(s)
        outputoriginal = os.path.normpath(outputfileoriginal[:-3] + outputformat)
        outputreconstructed = os.path.normpath(outputfilereconstructed[:-3] + outputformat)
        
        if(exportchoice == 'both'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputoriginal, point_cloud, delimiter = ',')
                np.savetxt(outputreconstructed, point_cloud_reconstructed, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputoriginal, point_cloud)
                np.save(outputreconstructed, point_cloud_reconstructed)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')  
        elif(exportchoice == 'original'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputoriginal, point_cloud, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputoriginal, point_cloud)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')
        elif(exportchoice == 'reconstructed'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputreconstructed, point_cloud_reconstructed, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputreconstructed, point_cloud_reconstructed)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')
        else:
            raise ValueError('Incorrect export choice. Choose \'both\', \'original\', or \'reconstructed\'.') 
    
class CSPCdft:
    """ --------------- The cspc (CSPCdft) library class. --------------- """
    ''' 
    A CSPCdft object consists of the following attributes/instance fields:
        - x     (x coordinate)
        - y     (y coordinate)
        - z     (z coordinate)
        - x_r   (reconstructed x coordinate)
        - y_r   (reconstructed y coordinate)
        - z_r   (reconstructed z coordinate)
        - k     (sparsity of basis-transformed point cloud)
        - basis_type (basis used for sparse transformation)
        - x_m (measured x coordinate basis coefficients)
        - y_m (measured y coordinate basis coefficients)
        - z_m (measured z coordinate basis coefficients)
        ??? - measurement_type (matrix used for measurememt matrix)
        ??? - CS_ratio
    '''
    
    # def __init__(self):
            
    def setLasCoords(self, pcpath, scaled = True):
        """ This method extracts the x,y,z coords from a .las file and
        fils up a CSPCdft object with these coords. """
        """ By default, this method extracts the scaled values because scaled values
        are better for compressed sensing. """
        
        ''' TO DO ___________________'''
        ''' Convert the self.x,.y,.z instance fields to np arrays rather than
        incompatible laspy.point.dims.ScaledArrayView type'''
        las = laspy.read(os.path.normpath(pcpath))
        # Scaled values
        if(scaled):
            self.x = np.array(las.x)
            self.y = np.array(las.y)
            self.z = np.array(las.z)
        # Raw values
        else:
            self.x = np.array(las.X)
            self.y = np.array(las.Y)
            self.z = np.array(las.Z)
         
    def setManualCoords(self, x, y, z):  
        """This method accepts 1D numpy arrays for each of the x,y,z coords
        and fills up a CSPCdft object with these coords. """ 
        self.x = x
        self.y = y
        self.z = z
       
    def showPC(self, name='Point Cloud'):
        """ This method plots the CSPCdft object with Open3D
        as a separate window. """
        # Take each coordinate and arrange each dimension by columns:
        point_data = np.stack([self.x, self.y, self.z], axis=0).transpose((1, 0)) 
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        o3d.visualization.draw_geometries([geom], window_name=name) # Show point cloud
        # Note: Hit "r" to reset PC view in Open3D
    
    def showReconstructedPC(self, name='Reconstructed Point Cloud'):
        """ This method plots the CSPCdft object with Open3D
        as a separate window from its reconstructed coordinates. """  
        # Take each coordinate and arrange each dimension by columns:
        point_data = np.stack([self.x_r, self.y_r, self.z_r], axis=0).transpose((1, 0)) 
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        o3d.visualization.draw_geometries([geom], window_name=name) # Show point cloud
        # Note: Hit "r" to reset PC view in Open3D
        return 0
    
    def writeLas(self, path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_original.las', las_point_format = 3, las_version = "1.2"):
        """ "This method stores the CSPCdft object's coordinates' as a .las file. """
        header = laspy.LasHeader(version=las_version, point_format=las_point_format)
        las = laspy.LasData(header)
        las.x = self.x
        las.y = self.y
        las.z = self.z
        las.write(os.path.normpath(path))
        
    def writeReconstructedLas(self, path = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_reconstructed.las', las_point_format = 3, las_version = "1.2"):
        """ "This method stores the CSPCdft object's reconstructed coordinates' as a .las file. """
        header = laspy.LasHeader(version=las_version, point_format=las_point_format)
        las = laspy.LasData(header)
        las.x = self.x_r
        las.y = self.y_r
        las.z = self.z_r
        las.write(os.path.normpath(path))       
    
    def performDownsample(self, dsrate, showPC=True, dstype='uniform'):
        """ This method performs uniform downsampling from Open3D 
            on the CSPCdft object. """
        """
        - Accepts x,y,z coordinate 1D ndarrays
        - Returns coordinates as (m x 3) matrix and shape
        - k is the downsampling rate (i.e. if k=2, 1/2 of the samples are kept).
        """
        # An open3D object has to be created in order to downsample
        point_data = np.stack([self.x, self.y, self.z], axis=0).transpose((1, 0)) # Take each coordinate in .las file and arrange it in an nd array
        geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
        geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
        
        # Downsampling function
        if(dstype=='uniform'):
            down_geom = geom.uniform_down_sample(dsrate)
        elif(dstype=='voxel'):
            down_geom = geom.voxel_down_sample(voxel_size=0.05)
        else:
            raise(ValueError("Invalid downsampling type. Select \"uniform\" or \"voxel\"."))
        
        ''' WHEN VOXEL DOWNSAMPLING IS DONE, THE OUTPUT IS LIST OF MATRICES AND NOT 1 MATRIX '''
        
        # Show point cloud
        if(showPC):
            o3d.visualization.draw_geometries([down_geom], window_name='Downsampled Point Cloud')
            
        np_points = np.asarray(down_geom.points) # Returns tuple of coordinate matrix and shape
        #dsshape = np_points[1]
        self.x = np_points[:,0] 
        self.y = np_points[:,1]
        self.z = np_points[:,2]
        #return dscoords, np_points
        #return np_points, np_points.shape # Returns matrix where each columns is downsampled coordinate for each dimension
    
    def transformDFT1D(self, padding=False):
        """ This method applies 1D DFT and returns DFT coefficients with padding if needed,
         for each dimension of the CSPCdft object. """
        coeffs_x = spfft.fft(self.x)
        coeffs_y = spfft.fft(self.y)
        coeffs_z = spfft.fft(self.z)
        
        """
        if(padding):
            # Ensure consistent size by padding to the next even size
            padded_length_x = (len(flat_coeffs_x) + 1) if len(flat_coeffs_x) % 2 != 0 else len(flat_coeffs_x)
            flat_coeffs_x = np.pad(flat_coeffs_x, (0, padded_length_x - len(flat_coeffs_x)))
            # Ensure consistent size by padding to the next even size
            padded_length_y = (len(flat_coeffs_y) + 1) if len(flat_coeffs_y) % 2 != 0 else len(flat_coeffs_y)
            flat_coeffs_y = np.pad(flat_coeffs_y, (0, padded_length_y - len(flat_coeffs_y)))
            # Ensure consistent size by padding to the next even size
            padded_length_z = (len(flat_coeffs_z) + 1) if len(flat_coeffs_z) % 2 != 0 else len(flat_coeffs_z)
            flat_coeffs_z = np.pad(flat_coeffs_z, (0, padded_length_z - len(flat_coeffs_z)))
        """  
        return coeffs_x, coeffs_y, coeffs_z
 
    def applyThresholding(arr, threshold):
        """" This method applies thresholding to the sparse signal before measurement to make even sparser. """
        """ Or this is done in place of the measurement matrix once the signal is in the sparse domain. """
        # calculateSparsity()
        # perform thresholding here to improve sparsity
        
        # For elements that are greater than or equal to the threshold, keep them the same
        # For elements that are < the threshold, replace them with zeros
        thresholded_arr = np.where(arr >= threshold, arr, 0)
        return thresholded_arr

    @staticmethod
    def transformIDFT1D(coeffs):
        """ This method accepts the DFT coeffs of a 1D signal/dimension, 
        and reconstructs based on the DFT coefficients. """
        # flat_coeffs: the flattened ndarray of the multi-level discrete Cosine Transformation coefficients for one of the PC's dimensions.
        # wavelet: the string of the chosen Wavelet to use for the DWC

        reconstructed = spfft.ifft(coeffs)
        return reconstructed.real
    
    def reconstructCVXPY(self, measured_x, measured_y, measured_z, phi):
        """ This method performs L1 minimization using the CVXPY library. """
        """ In order for each of the 3 dimensions to be reconstructed
        independently, the problem is solved 3 times with 3 different parameters."""
        """ Returns the reconstructed (x,y & z) coordinates along with time
        taken for reconstruction algorithm."""
        
        n = phi.shape[1] # Amount of coefficients
        
        # Define the optimization variable (the recovered DWT coefficients)
        s = cp.Variable(n, complex=True)
        # Reusable parameters
        if(measured_x.size == measured_y.size == measured_z.size):
            y = cp.Parameter(measured_x.size, complex=True)  
        else:
            raise ValueError("Point Cloud dimensions are mismatched in length. Unable to define \"y\".")

        # Define the L1-norm objective and constraint (y = Phi * x) and problem
        objective = cp.Minimize(cp.norm1(s))  # L1 minimization
        constraints = [phi @ s == y]
        prob = cp.Problem(objective, constraints)

        start_time = time()
        
        # Solve the optimization problems for x coordinates
        y.value = measured_x
        prob.solve(verbose = True)
        x_sparse_r = s.value # Sparse reconstructed value
        
        # Solve the optimization problems for y coordinates
        y.value = measured_y
        prob.solve(verbose = True)
        y_sparse_r = s.value # Sparse reconstructed value
        
        # Solve the optimization problems for z coordinates
        y.value = measured_z
        prob.solve(verbose = True)
        z_sparse_r = s.value # Sparse reconstructed value
        
        # Time of optimization algorithm
        end_time = time()
        solve_time = end_time - start_time

        if prob.status != cp.OPTIMAL:
            raise ValueError(f"Optimization failed: {prob.status}")

        # Inverse transform back to original domain
        self.x_r = CSPCdft.transformIDFT1D(x_sparse_r) #TODO
        self.y_r = CSPCdft.transformIDFT1D(y_sparse_r) #TODO
        self.z_r = CSPCdft.transformIDFT1D(z_sparse_r) #TODO
        
        print(f"Solver Time: {solve_time} [s]")
        return solve_time

    def reconstructCVXPY_ray(self, measured_x, measured_y, measured_z, phi):
        """ This method performs L1 minimization using the CVXPY library. """
        """ In order for each of the 3 dimensions to be reconstructed
        independently, the problem is solved 3 times with 3 different parameters."""
        """ Returns the reconstructed (x,y & z) coordinates along with time
        taken for reconstruction algorithm."""

        ray.init(ignore_reinit_error=True)

        start_time = time()
        
        @ray.remote(num_cpus=2)
        def reconstruct(measured_coord, phi):
            # Amount of coefficients to reconstruct:
            n = phi.shape[1]
            # Define the optimization variable (the recovered DWT coefficients:
            s = cp.Variable(n)
            # Reusable parameter:
            y = cp.Parameter(len(measured_coord))
            y.value = measured_coord
            # Define the L1-norm objective and constraint (y = Phi * x) and problem
            objective = cp.Minimize(cp.norm1(s))
            constraints = [phi @ s == y]
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=True)
            return s.value # Return sparse reconstructed value
        
        # NOTE: Using Ray for large objects, like large arrays can cause performance issues!
        # So, I am using ray.put() to store the large object in the Ray object store.
        measured_x_ref = ray.put(measured_x)
        measured_y_ref = ray.put(measured_y)
        measured_z_ref = ray.put(measured_z)
        phi_ref = ray.put(phi)
        
        # Parallel processing reconstruction:
        future_x = reconstruct.remote(measured_x_ref, phi_ref)
        future_y = reconstruct.remote(measured_y_ref, phi_ref)
        future_z = reconstruct.remote(measured_z_ref, phi_ref)
        result_x_sparse_r = ray.get(future_x)
        result_y_sparse_r = ray.get(future_y)
        result_z_sparse_r = ray.get(future_z)
        
        # Time of optimization algorithm
        end_time = time()
        solve_time = end_time - start_time
        
        # Inverse transform back to original domain
        self.x_r = CSPCdft.transformIDFT1D(result_x_sparse_r) #TODO
        self.y_r = CSPCdft.transformIDFT1D(result_y_sparse_r) #TODO
        self.z_r = CSPCdft.transformIDFT1D(result_z_sparse_r) #TODO
        
        print(f"Solver Time: {solve_time} [s]")
        return solve_time

    def reconstructCosamp(): #TODO
        pass

    def calculateReconstructionError(self):
        """ This method calculates and outputs the 2-Norm Error, MSE Error, RMSE Error, and MAE error of the Point Cloud object 
        to the console and returns them as a dictionary. """
        
        point_cloud = np.column_stack((self.x, self.y, self.z))
        point_cloud_reconstructed = np.column_stack((self.x_r, self.y_r, self.z_r))
        
        l2error_val = np.linalg.norm(point_cloud - point_cloud_reconstructed) / np.linalg.norm(point_cloud)
        print("Reconstruction Errors ------------------------- \n")
        print(f"2-Norm Error: {l2error_val:.4f}")
        # Mean Squared Error
        mse_val = mean_squared_error(point_cloud, point_cloud_reconstructed)
        print("MSE: ", mse_val)
        # Root Mean Squared Error
        rmse_val = math.sqrt(mse_val)
        print("RMSE: ", rmse_val)
        # Mean Absolute Error
        mae_val = mean_absolute_error(point_cloud, point_cloud_reconstructed)
        print("MAE: ", mae_val)
        
        # Chamfer Distance
        cd_val = pcu.chamfer_distance(point_cloud, point_cloud_reconstructed)
        print("Chamfer Distance: ", cd_val)
        
        # Hausdorff Distance
        hd_val = pcu.hausdorff_distance(point_cloud, point_cloud_reconstructed)
        print("Hausdorff Distance: ", hd_val)
        
        # Earth Mover's Distance
        emd_val = 0 # TODO Disable EMD for now since it takes too long to calculate
        #emd_val, pi_val = pcu.earth_movers_distance(point_cloud, point_cloud_reconstructed) # Calculating EMD takes a VERY long time
        print("Earth Mover's Distance: ", emd_val)

        errors = dict(l2norm = l2error_val, MSE = mse_val, RMSE = rmse_val, MAE = mae_val, CD = cd_val, HD = hd_val, EMD = emd_val)
        
        return errors

    def plotPCs(self, main_title, e_angle=230, a_angle=-240, outputfile='D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_reconstructed.las', fileformat='pdf', heightshown=True, pointsize=5, colormap='cool'):
        """ Plots the original and reconstructed point clouds and saves the plot as .pdf, .jpg, .png, or .svg. """
        """ Expects the output file to be the original output .las file path. """
        
        # Changes to .pdf file (or other specified formats: .png, .jpg, .svg) from .las
        outputpath = os.path.normpath(outputfile[:-3] + fileformat)
        
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(main_title)
        
        if(colormap == 'BGYR'):
            blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
            colormap = blue_green_yellow_red
        
        if(heightshown == True): # With elevation colors
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.x, self.y, self.z, c=self.z, cmap=colormap, s=pointsize)
            ax1.set_title("Original Point Cloud")
            ax1.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax1.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax1.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax1.set_xlabel('$X$', fontsize=15)
            ax1.set_ylabel('$Y$', fontsize=15)
            ax1.set_zlabel('$Z$', fontsize=15)
    
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(self.x_r, self.y_r, self.z_r, c=self.z, cmap=colormap, s=pointsize)
            ax2.set_title("Reconstructed Point Cloud")
            ax2.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax2.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax2.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax2.set_xlabel('$X$', fontsize=15)
            ax2.set_ylabel('$Y$', fontsize=15)
            ax2.set_zlabel('$Z$', fontsize=15)
        else: # Without elevation values
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.x, self.y, self.z, c='b' , s=pointsize)
            ax1.set_title("Original Point Cloud")
            ax1.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax1.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax1.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax1.set_xlabel('$X$', fontsize=15)
            ax1.set_ylabel('$Y$', fontsize=15)
            ax1.set_zlabel('$Z$', fontsize=15)
    
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(self.x_r, self.y_r, self.z_r, c='b', s=pointsize)
            ax2.set_title("Reconstructed Point Cloud")
            ax2.set_xlim(math.floor(min(self.x)), math.ceil(max(self.x)))
            ax2.set_ylim(math.floor(min(self.y)), math.ceil(max(self.y)))
            ax2.set_zlim(math.floor(min(self.z)), math.ceil(max(self.z)))
            ax2.set_xlabel('$X$', fontsize=15)
            ax2.set_ylabel('$Y$', fontsize=15)
            ax2.set_zlabel('$Z$', fontsize=15)
        
        ax1.view_init(elev=e_angle, azim=a_angle)
        ax2.view_init(elev=e_angle, azim=a_angle)

        plt.savefig(outputpath)
        plt.show()

    def exportCoords(self, outputfileoriginal='D:\Documents\Thesis_CS\Point_Cloud_Inputs\original.las', outputfilereconstructed='D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_reconstructedcoords.las', outputformat='csv', exportchoice='both'):
        """ This method saves and exports the coordinates of both the original point cloud and the reconstructed point cloud
        into .csv or .npy file formats. """
        # Convert .las point clouds to a (n x 3) numpy array
        point_cloud = np.column_stack((self.x, self.y, self.z))
        point_cloud_reconstructed = np.column_stack((self.x_r, self.y_r, self.z_r))
        # Paths for exporting the npy file(s)
        outputoriginal = os.path.normpath(outputfileoriginal[:-3] + outputformat)
        outputreconstructed = os.path.normpath(outputfilereconstructed[:-3] + outputformat)
        if(exportchoice == 'both'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputoriginal, point_cloud, delimiter = ',')
                np.savetxt(outputreconstructed, point_cloud_reconstructed, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputoriginal, point_cloud)
                np.save(outputreconstructed, point_cloud_reconstructed)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')  
        elif(exportchoice == 'original'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputoriginal, point_cloud, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputoriginal, point_cloud)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')
        elif(exportchoice == 'reconstructed'):
            if(outputformat == 'csv'): # Save as csv
                np.savetxt(outputreconstructed, point_cloud_reconstructed, delimiter = ',')
            elif(outputformat == 'npy'): # Save as npy
                np.save(outputreconstructed, point_cloud_reconstructed)
            else:
                raise ValueError('Incorrect output format specified. Choose \'csv\' or \'npy\'.')
        else:
            raise ValueError('Incorrect export choice. Choose \'both\', \'original\', or \'reconstructed\'.')    

    
    
def setupParameters(path, lasfile, num_points, cs_ratio, measurement_type, pcname, basis, wvlt, sparsity, ds_type='none'):
    """ 
    This function generates an output path/filename, a plot title, and text description of Point Cloud Compressed Sensing parameters.
    """
    if (basis != 'DWT'):
        wvlt = '' # Ignore wvlt string if not DWT
        basis_file_str = f'{basis}' # Removes extra underscore
    elif((basis == 'DWT') and wvlt == ''):
        raise ValueError(f'Wavelet not specified for {basis}!')
    else:         
        basis_file_str = f'{basis}_' + wvlt.capitalize() # Wavelet type string interpolation for file name
    
    ds_file_str = 'ds_' + ds_type # Downsampling description for output file name
    if(ds_type == 'uniform'):
        ds_str = '(Uniformly Downsampled)'
    elif(ds_type == 'voxel'):
        ds_str = '(Voxel Downsampled)'
    elif(ds_type == 'none'):
        ds_str = ''
        ds_file_str = ''
    else:
        raise ValueError('Invalid type of downsampling! If no downsampling was performed, set ds_type to \'none\'.')
    
    lasfilename = lasfile.split('.')[0] # Removes .las extension to be used for naming later
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    basis_txt_str = wvlt.capitalize() + f' {basis}' # Wavelet type string interpolation for plot title and text description
    measurement_type_str = measurement_type.capitalize()
    
    outputpath = specifyOutputPath(path, lasfilename, f'{sparsity}thresholded_{basis_file_str}_{cs_ratio_str}{measurement_type_str}_of_{num_points}{ds_file_str}.las')  
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point {pcname} PC Using {basis_txt_str}"
    metadata = f"{cs_percentage_str}% CS Ratio | {num_points} points {ds_str} | {pcname} PC | {measurement_type_str} Measurement | {basis_txt_str} | {sparsity}% Sparse | CVXPY CLARABEL Reconstruction"

    return outputpath, plot_title, metadata

def setupSimulationParameters(path, lasfile, num_points, cs_ratio, measurement_type, pcname, basis, wvlt, ds_type='none', i=1000):
    """ 
    This function generates an output path/filename, a plot title, and text description of Point Cloud Compressed Sensing parameters and simulation
    output results for a txt file.
    """
    if (basis != 'DWT'):
        wvlt = '' # Ignore wvlt string if not DWT
        basis_file_str = f'{basis}' # Removes extra underscore
    elif((basis == 'DWT') and wvlt == ''):
        raise ValueError(f'Wavelet not specified for {basis}!')
    else:         
        basis_file_str = f'{basis}_' + wvlt.capitalize() # Wavelet type string interpolation for file name
    
    ds_file_str = 'ds_' + ds_type # Downsampling description for output file name
    if(ds_type == 'uniform'):
        ds_str = '(Uniformly Downsampled)'
    elif(ds_type == 'voxel'):
        ds_str = '(Voxel Downsampled)'
    elif(ds_type == 'none'):
        ds_str = ''
        ds_file_str = ''
    else:
        raise ValueError('Invalid type of downsampling! If no downsampling was performed, set ds_type to \'none\'.')
    
    lasfilename = lasfile.split('.')[0] # Removes .las extension to be used for naming later
    cs_ratio_str = str(format(cs_ratio, ".2f"))[2:] # Measurement ratio string interpolation
    cs_percentage_str = str(int(cs_ratio * 100)) # Measurement percentage string interpolation
    basis_txt_str = wvlt.capitalize() + f' {basis}' # Wavelet type string interpolation for plot title and text description
    measurement_type_str = measurement_type.capitalize()
    
    outputpath = specifyOutputPath(path, lasfilename, f'{basis_file_str}_{cs_ratio_str}{measurement_type_str}_of_{num_points}{ds_file_str}_{i}simulations.txt')  
    plot_title = f"{cs_percentage_str}% Compression of {num_points}-Point {pcname} PC Using {basis_txt_str}" #TODO: Do not really need this
    metadata = f"{i}-Iteration Simulation Results For {cs_percentage_str}% CS Ratio | {num_points} points {ds_str} | {pcname} PC | {measurement_type_str} Measurement | {basis_txt_str} | CVXPY ECOS (Embedded Conic Solver) Reconstruction"

    return outputpath, plot_title, metadata
    
def pcToArray(pcpath):
    """ This function converts a .las point cloud to an (m x 3) matrix. """
    las = laspy.read(os.path.normpath(pcpath))
    '''print(las.x)
    print(las.y)
    print(las.z)'''
    return np.stack([las.x,las.y,las.z], axis=-1)
    #return np.array()
    
def arrayToPC(pcarray, pctype='dwt'):
    """ This function converts an (m x 3) matrix to a .las point cloud."""
    if(pctype=='dwt'):
        pointcloud = CSPCdwt()
        pointcloud.setManualCoords(pcarray[:,0], pcarray[:,1], pcarray[:,2])
    elif(pctype=='dct'):
        pointcloud = CSPCdct()
        pointcloud.setManualCoords(pcarray[:,0], pcarray[:,1], pcarray[:,2])
    elif(pctype=='dft'):
        pointcloud = CSPCdft()
        pointcloud.setManualCoords(pcarray[:,0], pcarray[:,1], pcarray[:,2])
    else:
        raise ValueError('Invalid PC Type. Choose "dwt", "dct", or "dft"')
    return pointcloud

def pclength(pcpath):
    """
    This function returns the length of the point cloud from the .las file name string.
    """
    las = laspy.read(os.path.normpath(pcpath))
    # Scaled values
    if(len(las.x) == len(las.y) == len(las.z)):
        return len(las.x)
    else:
        raise ValueError('Point Cloud dimensions do not match!')

def calculateSparsity(x_transformed, y_transformed, z_transformed, threshold=None):
    """" This method calculates sparsity of the basis-transformed signal and also returns the number of nonzero coefficients. """
    """  If the threshold variable is set to None, the sparsity is just calculated from zero values.
    If the threshold variable is set to any value, the sparsity calculation counts any values below
    or at the threshold as a "zero". """
    """ Sparsity value = 0 ---> No zero values """
    """ Sparsity value = 1 ---> Only zero (or close to zero) values """
    # Any value below the threshold is counted as a "zero"
    # If there is no threshold set, just count zeros
    if (threshold == None):
        x_sparsity = round(1.0 - (np.count_nonzero(x_transformed) / float(x_transformed.size)), 4)
        y_sparsity = round(1.0 - (np.count_nonzero(y_transformed) / float(y_transformed.size)), 4)
        z_sparsity = round(1.0 - (np.count_nonzero(z_transformed) / float(z_transformed.size)), 4)
        sparsity = dict(x=x_sparsity, y=y_sparsity, z=z_sparsity)  
        print("Sparsity values ------------------------- \n")
        print(f"x: {x_sparsity}\n")
        print(f"y: {y_sparsity}\n")
        print(f"z: {z_sparsity}\n")
        x_s = np.count_nonzero(x_transformed)
        y_s = np.count_nonzero(y_transformed)
        z_s = np.count_nonzero(z_transformed)
        s_sparse = dict(x=x_s, y=y_s, z=z_s)
        return sparsity, s_sparse # Round to 4 decimal places
    elif (threshold > 0):
        x_sparsity = round(1.0 - (np.count_nonzero(x_transformed > threshold) / float(x_transformed.size)), 4)
        y_sparsity = round(1.0 - (np.count_nonzero(y_transformed > threshold) / float(y_transformed.size)), 4)
        z_sparsity = round(1.0 - (np.count_nonzero(z_transformed > threshold) / float(z_transformed.size)), 4)
        sparsity = dict(x=x_sparsity, y=y_sparsity, z=z_sparsity)
        print("Sparsity values ------------------------- \n")
        print(f"x: {x_sparsity}\n")
        print(f"y: {y_sparsity}\n")
        print(f"z: {z_sparsity}\n")
        x_s = np.count_nonzero(x_transformed > threshold)
        y_s = np.count_nonzero(y_transformed > threshold)
        z_s = np.count_nonzero(z_transformed > threshold)
        s_sparse = dict(x=x_s, y=y_s, z=z_s)
        return sparsity, s_sparse
    else:
        raise(ValueError("threshold must be greater than zero. If no threshold is specified, zeros are counted instead."))
 
def plotCoefficients(fcx, fcy, fcz):
    """ This method plots the flattened DWT coefficients for the x, y, and z coordinates. """
    fig_coeffs = plt.figure(num="Coeffs Plot")
         
    fig, axs = plt.subplots(1, 3)

    # Plot data on the first subplot
    axs[0].plot(fcx)
    axs[0].set_title('x DWT')
        
    # Plot data on the second subplot
    axs[1].plot(fcy)
    axs[1].set_title('y DWT')
        
    # Plot data on the third subplot
    axs[2].plot(fcz)
    axs[2].set_title('z DWT')
    # Adjust the spacing between subplots
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()
 
def applyThresholding(arr, percentile, type_th='hard'):
    """" This method applies thresholding to the sparse signal before measurement to make even sparser. """
    """ Or this is done in place of the measurement matrix once the signal is in the sparse domain. """
    """
    percentile: Percentage of top-valued coefficients to keep
    (100 - percentile) would be the sparsity percentage
    """
    # calculateSparsity()
    # perform thresholding here to improve sparsity
    
    # Percentile-based thresholding:
        # reduce_percentage means: Reduce X% amount of lower values to zero
        # For example: If reduce_percentage = 90%, only the 10% highest values are kept
        # percentile means: percentage of highest values that are kept
    reduce_percentage = 100 - percentile
    threshold = np.percentile(np.abs(arr), reduce_percentage)
    
    # For hard thresholding:
        # For elements that are greater than the threshold, keep them the same
        # For elements that are <= the threshold, replace them with zeros
    # For soft thresholding:
        # Sets small coefficients to zero, but shrinks larger coefficients to reduce noise smoothly.
    if(type_th =='hard'): # Hard thresholding
        thresholded_arr = np.stack([np.where(np.abs(a) > threshold, a, 0) for a in arr])
    elif(type_th =='soft'): # Soft thresholding
        thresholded_arr = np.stack([np.sign(a) * np.maximum(np.abs(a) - threshold, 0) for a in arr])
    else:
        raise ValueError('Incorrect thresholding type specified! Only \'hard\' thresholding or \'soft\' thresholding are allowed.')
    
    return thresholded_arr, threshold
    
def generateMeasurementMatrix(m, n, type='gaussian'):
    """ Generates a random measurement matrix of size m x n,
    where m is the subsampled length of values and n is the original 
    length of the signal (or its coefficients). """
    if(type == 'gaussian'):
        Phi = np.random.randn(m, n) / np.sqrt(m)
    elif(type == 'gaussian_normal'): # Gaussian distribution from normalized standard normal distribution
        Phi = np.random.randn(m, n)
        Phi /= np.linalg.norm(Phi, axis=0)  # Normalize columns
    elif(type == 'bernoulli_symmetric'):
        Phi = np.random.choice([-1, 1], size=(m, n))
        #Phi = Phi / np.sqrt(m)  # Scale by sqrt(m) for proper normalization
    elif(type == 'bernoulli_standard'): 
        Phi = np.random.choice([0, 1], size=(m, n))
    elif(type == 'gaussian2'):
        Phi = np.random.normal(0, 1, (m, n))
    else:
        raise ValueError("Invalid Measurement Type. Select \'gaussian\', \'gaussian normal\', or \'bernoulli\'.")
    return Phi
    
def measure1D(Phi, x_flat_coeffs, y_flat_coeffs, z_flat_coeffs):
    """ This method takes compressed measurements of each sparse-represented 
    dimension of the CSPCdwt object: y = Phi @ signal[:][0]. """
    measured_x = Phi @ x_flat_coeffs
    measured_y = Phi @ y_flat_coeffs
    measured_z = Phi @ z_flat_coeffs
    return measured_x, measured_y, measured_z
     
def coherence(Phi, Psi):
    """Compute coherence between measurement matrix Phi and basis matrix Psi."""
    A = np.dot(Phi, Psi)  # Compute sensing matrix A = Phi * Psi
    A_normed = A / np.linalg.norm(A, axis=0, keepdims=True)  # Normalize columns
    np.fill_diagonal(A_normed, 0)  # Ignore self-coherence
    return np.max(np.abs(A_normed))  # Find max absolute off-diagonal value

def downsample(inputpath, lasfile, ds_percentage=3e-4, show=False, typeds='uniform', outputpath='D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled'):
    """ Downsamples .las file and exports the downsampled version as a .las file."""
    """ Returns the downsampled file path and the size of the downsampled point cloud."""
    """ NOTE: (ds_percentage * original number of points) MUST be an odd value."""
    pc_ds = CSPCdct() #TODO
    pc_ds.setLasCoords(os.path.normpath(os.path.join(inputpath, lasfile))) #TODO
    # Downsampling
    dsrate = int(1/ds_percentage) # Keep every "dsrate" amount of points
    pc_ds.performDownsample(dsrate, showPC=show, dstype=typeds) #TODO
    num_points = pc_ds.x.size # Amount of resulting points after downsampling
    outputlasfile = lasfile[:-4] + f'_{num_points}ds_{typeds}' + lasfile[-4:] #TODO
    ds_path = os.path.normpath(os.path.join(outputpath, outputlasfile))
    if(num_points % 2 != 0):
        raise ValueError(f"Number of points has to be even after downsampling ({num_points} points)! Adjust ds_percentage.")
    pc_ds.writeLas(path=ds_path) #TODO
    return ds_path, num_points
  
def voxelizePC():
    print('input')
    armadillo = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(armadillo.path)

    N = 2000
    pcd = mesh.sample_points_poisson_disk(N)
    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    o3d.visualization.draw_geometries([pcd])

    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])
    
def voxelDownsample():
    n_points = 25000
    dsrate = 2
    x = 10*np.exp(np.linspace(0, 5, n_points))+1000*np.random.rand(n_points) 
    y = 2500*np.sin(np.linspace(1, 5000*np.pi, n_points))+250*np.random.rand(n_points) 
    z = np.linspace(1, 1000, n_points)
    
    point_data = np.stack([x,y,z], axis=0).transpose((1, 0)) # Take each coordinate in .las file and arrange it in an nd array
    geom = o3d.geometry.PointCloud() # Create Open3D point cloud object
    geom.points = o3d.utility.Vector3dVector(point_data) # Converts float 64 numpy array of shape (n,3) to Open3D format and store it into "points" attribute
    
    # Downsampling function
    down_geom = geom.uniform_down_sample(dsrate)
    down_geom_voxel = geom.voxel_down_sample(voxel_size=1.5)
    
    
    
    o3d.visualization.draw_geometries([down_geom], window_name='Downsampled Point Cloud')
    o3d.visualization.draw_geometries([down_geom_voxel], window_name='Downsampled Voxel Point Cloud')
    
def exportReconstructionInfo(info, errors, solve_time, sparsity, outputfile='D:/Documents/Thesis_CS/Point_Cloud_Outputs/output_reconstructed.las', ):
    """ This function exports any detailed reconstruction info to a .txt file """
    """ Expects the output file to be the original output .las file path. """
        
    # Changes to .txt file from .las
    outputpathtxt = os.path.normpath(outputfile[:-3] + 'txt')
    filetxt = open(outputpathtxt, "w")
        
    filetxt.write(info)
    filetxt.write("\n \n--- ")
    filetxt.write("Reconstruction Errors ------------------------- \n")
    filetxt.write(f"2-Norm Error: {errors['l2norm']:.4f} \n")
    # Mean Squared Error
    filetxt.write(f"Mean Squared Error: {errors['MSE']:.4f} \n")
    # Root Mean Squared Error
    filetxt.write(f"Root Mean Squared Error: {errors['RMSE']:.4f} \n")
    # Mean Absolute Error
    filetxt.write(f"Mean Absolute Error: {errors['MAE']:.4f} \n")
    # Chamfer Distance
    filetxt.write(f"Chamfer Distance: {errors['CD']:.4f} \n")
    # Hausdorff Distance
    filetxt.write(f"Hausdorff Distance: {errors['HD']:.4f} \n")
    # Earth Mover's Distance
    filetxt.write(f"Earth Mover's Distance: {errors['EMD']:.4f} \n")
    filetxt.write("\n")
    
    # Solver Time
    filetxt.write(f"Solver Time: {solve_time} [s]\n")
    filetxt.write("\n")
    # TODO: These sparsity values are not relevant anymore, REMOVE 
    filetxt.write("Sparsity values:\n")
    filetxt.write(f"        x: {sparsity['x']}\n")
    # Solver Time
    filetxt.write(f"        y: {sparsity['y']}\n")
    # Solver Time
    filetxt.write(f"        z: {sparsity['z']}\n")
        
    filetxt.close()

# TODO UNUSED FUNCTION, MAY BE NOT USEFUL
# def exportSimulationInfo(outputpath, metadata, l2norm_arr, MSE_arr, RMSE_arr, MAE_arr, solvertime_arr):
#     """ This function exports any simulation info to a .txt file. """
#     avg_l2norm = np.mean(l2norm_arr)
#     avg_MSE = np.mean(MSE_arr)
#     avg_RMSE = np.mean(RMSE_arr)
#     avg_MAE = np.mean(MAE_arr)
#     avg_solvertime = np.mean(solvertime_arr)
    
#     filetxt = open(os.path.normpath(outputpath), "w")
#     filetxt.write(metadata)
#     filetxt.write("\n \n--- \n\n")
#     filetxt.write(f"Average 2-Norm Error: {avg_l2norm:.4f} \n")
#     # Mean Squared Error
#     filetxt.write(f"Average Mean Squared Error: {avg_MSE:.4f} \n")
#     # Root Mean Squared Error
#     filetxt.write(f"Average Root Mean Squared Error: {avg_RMSE:.4f} \n")
#     # Mean Absolute Error
#     filetxt.write(f"Average Mean Absolute Error: {avg_MAE:.4f} \n")
#     # Solver Time
#     filetxt.write(f"Average Solver Time: {avg_solvertime:.4f} [s]\n")
#     filetxt.write("\n")
#     filetxt.write("-----------------------------------")
#     filetxt.write("\n2-Norm:\n")
#     for element in l2norm_arr:
#         filetxt.write(str(element) + ',')
#     filetxt.write("\n\nMSE:\n")
#     for element in MSE_arr:
#         filetxt.write(str(element) + ',')
#     filetxt.write("\n\nRMSE:\n")
#     for element in RMSE_arr:
#         filetxt.write(str(element) + ',')
#     filetxt.write("\n\nMAE:\n")
#     for element in MAE_arr:
#         filetxt.write(str(element) + ',')
#     filetxt.write("\n\nSolver times:\n")
#     for element in solvertime_arr:
#         filetxt.write(str(element) + ',')        
#     filetxt.close()

def specifyOutputPath(output_folderpath, output_PCname, output_PCdescription):
    """ 
    This function returns a path to your point cloud .las output.
        - output_folderpath: Path to point cloud output directory (without point cloud name).
        - output_PCname: Name of output .las point cloud without .las extension.
        - output_PCdescription: Second part of name of output .las point cloud (meant for describing basis, CS ratio, and number of points).
            RETURNS (str): output_folderpath + output_PCname + output_PCdescription.
            (i.e. 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/helix' +  'helix_reconstructed' + 'haar_0.50m_of_5000ds_uniform.las')
    """
    path_filename = output_PCname + '_' + output_PCdescription
    return os.path.normpath(os.path.join(output_folderpath, path_filename))
    
def plotPC(x, y, z, title, e_angle, a_angle, heightshown=True, pointsize=5, colormap='twilight'):
    """ This function provides a 3D scatter plot of x,y,z coordinates without
    belonging to any class or object. """
    fig = plt.figure(figsize=(12, 6))
        
    ax1 = fig.add_subplot(111, projection='3d')
    
    if(heightshown == True):
        if(colormap == 'BGYR'):
            blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
            colormap = blue_green_yellow_red
        ax1.scatter(x, y, z, c=z, cmap=colormap, s=pointsize)
        ax1.set_title(title)
        ax1.set_xlim(math.floor(min(x)), math.ceil(max(x)))
        ax1.set_ylim(math.floor(min(y)), math.ceil(max(y)))
        ax1.set_zlim(math.floor(min(z)), math.ceil(max(z)))
        ax1.set_xticks(np.linspace(min(x), max(x), 3)) # Fix overlapping of x labels
        ax1.set_yticks(np.linspace(min(y), max(y), 3)) # Fix overlapping of x labels
        ax1.set_xlabel('$X$', fontsize=15)
        ax1.set_ylabel('$Y$', fontsize=15)
        ax1.set_zlabel('$Z$', fontsize=15)
        ax1.view_init(elev=e_angle, azim=a_angle) # e-angle 
    else:
        if(colormap == 'BGYR'):
            blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
            colormap = blue_green_yellow_red
        ax1.scatter(x, y, z, c='b', cmap=colormap, s=pointsize)
        ax1.set_title(title)
        ax1.set_xlim(math.floor(min(x)), math.ceil(max(x)))
        ax1.set_ylim(math.floor(min(y)), math.ceil(max(y)))
        ax1.set_zlim(math.floor(min(z)), math.ceil(max(z)))
        ax1.set_xlabel('$X$', fontsize=15)
        ax1.set_ylabel('$Y$', fontsize=15)
        ax1.set_zlabel('$Z$', fontsize=15)
        ax1.view_init(elev=e_angle, azim=a_angle)
        
def plotContourPC(x, y, z, title='Filled Contour Plot', pointsize=25, colormap='plasma'):
    """ This function plots a top-down 3D contour plot to visualize elevation values with a colormap."""
    
    # Custom blue-green-yellow-red colormap
    if(colormap == 'BGYR'):
        blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
        colormap = blue_green_yellow_red     
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter plot (top-down view)
    sc = ax.scatter(x, y, c=z, cmap=colormap, edgecolor='none', s=pointsize)
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Elevation (Z values)')
    # Labels and title
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_title(title)
 
        