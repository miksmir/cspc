# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 06:13:34 2025

@author: Mikhail
"""

""" 
This script takes the (x,y,z) coordinates from an (n x 3) matrix point 
cloud and plots it either as a 3D scatter plot or as an overhead (top-down)
contour plot.
"""


import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors











# Custom colormap
c_dict = {
    'red':   [(0.0, 0.0, 0.0), (0.33, 0.0, 0.0), (0.66, 1.0, 1.0), (1.0, 1.0, 1.0)],
    'green': [(0.0, 0.0, 0.0), (0.33, 1.0, 1.0), (0.66, 1.0, 1.0), (1.0, 0.0, 0.0)],
    'blue':  [(0.0, 1.0, 1.0), (0.33, 1.0, 1.0), (0.66, 0.0, 0.0), (1.0, 0.0, 0.0)]
}



def plotPC(x, y, z, title, e_angle, a_angle, heightshown=True, pointsize=5, colormap='twilight'):
    """ This function provides a 3D scatter plot of x,y,z coordinates without
    belonging to any class or object. """
    fig = plt.figure(figsize=(12, 8))
        
    ax1 = fig.add_subplot(111, projection='3d')
    
    if(heightshown == True):
        if(colormap == 'BGYR'):
            blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
            colormap = blue_green_yellow_red
        ax1.scatter(x, y, z, c=z, cmap=colormap, s=pointsize)
        #ax1.set_title(title)
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
        #ax1.set_title(title)
        ax1.set_xlim(math.floor(min(x)), math.ceil(max(x)))
        ax1.set_ylim(math.floor(min(y)), math.ceil(max(y)))
        ax1.set_zlim(math.floor(min(z)), math.ceil(max(z)))
        ax1.set_xlabel('$X$', fontsize=15)
        ax1.set_ylabel('$Y$', fontsize=15)
        ax1.set_zlabel('$Z$', fontsize=15)
        ax1.view_init(elev=e_angle, azim=a_angle)
    #plt.savefig("C:/Users/misha/Downloads/ColumbusCircle_90thresholded_25DCT.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("C:/Users/misha/Downloads/MountMarcy_90thresholded_25Db2.pdf", format="pdf", bbox_inches="tight")
        
def plotContourPC(x, y, z, title='Filled Contour Plot', pointsize=25, colormap='plasma'):
    """ This function plots a top-down 3D contour plot to visualize elevation values with a colormap."""
    
    # Custom blue-green-yellow-red colormap
    if(colormap == 'BGYR'):
        blue_green_yellow_red = colors.LinearSegmentedColormap('blue_green_yellow_red', c_dict)
        colormap = blue_green_yellow_red     
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    # Scatter plot (top-down view)
    sc = ax.scatter(x, y, c=z, cmap=colormap, edgecolor='none', s=pointsize)
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Elevation (Z values) [m]')
    # Labels and title
    ax.set_xlabel('$X [m]$')
    ax.set_ylabel('$Y [m]$')
    ax.set_title(title)
    #plt.savefig("C:/Users/misha/Downloads/ColumbusCircle_90thresholded_25DCT_contour.pdf", format="pdf", bbox_inches="tight")
    plt.savefig("C:/Users/misha/Downloads/MountMarcy_90thresholded_25Db2_contour.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    pcname = 'Mount Marcy'
    
    #file = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.npy'
    #file = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_DCT_50Gaussian_of_5000ds_uniform.npy'
    
    #file = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_5000ds_uniform.npy'
    #file = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_90thresholded_DCT_25Gaussian_of_5000ds_uniform.npy'
    file = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/MtMarcy_u_5865088400_2015_90thresholded_DWT_Db2_25Gaussian_of_5000ds_uniform.npy'
    
    pc = np.load(file)
    
    #plot_title = f'Original: {pcname} ({pc.shape[0]} points)'
    plot_title = f'Reconstructed: {pcname} (25% Reconstructed Db2 DWT at 90% Sparsity)'
    
    
    plotPC(x=pc[:,0], y=pc[:,1], z=pc[:,2], title=plot_title, e_angle=35, a_angle=-205, colormap='BGYR')
    plotContourPC(pc[:,0], pc[:,1], pc[:,2], title=plot_title, colormap='BGYR')
    
    
    