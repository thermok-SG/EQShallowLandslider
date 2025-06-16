# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:32:53 2025

@author: tchales
"""

# %% Import packages

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize  # Import minimize from scipy.optimize
from scipy.stats import ks_2samp
 
# %%

def compare_inverse_gamma(data1, data2, continuous=True, binned=True):
     
    # Fit an inverse gamma distribution to the data
    shape1, loc1, scale1 = stats.invgamma.fit(data1, floc=0)  # Force location to 0
    shape2, loc2, scale2 = stats.invgamma.fit(data2, floc=0)
     
    # Display fitted parameters
    print(f"Dataset 1 - Fitted shape: {shape1}, scale: {scale1}")
    print(f"Dataset 2 - Fitted shape: {shape2}, scale: {scale2}")
     
    sorted_data1 = np.sort(data1)
    pdf1 = stats.invgamma.pdf(sorted_data1, shape1, loc1, scale1)
     
    sorted_data2 = np.sort(data2)
    pdf2 = stats.invgamma.pdf(sorted_data2, shape2, loc2, scale2)
    
    # Plot inv gamma for continuous data
    fig1 = plt.figure(figsize=(12,8), layout='constrained')
    plt.scatter(sorted_data1, pdf1, color='0.8', label='Roback et al. landslides', s=10)
    plt.plot(sorted_data1, pdf1, color='0.8')
     
    plt.scatter(sorted_data2, pdf2, color='0.2', label='Modelled landslides', s=10)
    plt.plot(sorted_data2, pdf2, color='0.2')
     
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-11, max(max(pdf1), max(pdf2))*5)
    
    plt.xlabel('Area')
    plt.title('Inverse gamma distribution')
    plt.legend()
    plt.show()
    
    # Inverse gamma for binned data
    # Create log-spaced bins
    bins = np.logspace(np.log10(min(data1.min(), data2.min())), np.log10(max(data1.max(), data2.max())), 30)
     
    # Count the number of data points in each bin (histogram)
    hist1, bin_edges1 = np.histogram(data1, bins=bins)
    hist2, bin_edges2 = np.histogram(data2, bins=bins)
     
    # Calculate bin centers
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
    bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
     
    # Normalize the histogram to represent the probability density
    hist1_normalized = hist1 / (np.sum(hist1) * np.diff(bin_edges1))  # PDF for dataset 1
    hist2_normalized = hist2 / (np.sum(hist2) * np.diff(bin_edges2))  # PDF for dataset 2
     
    # Define the negative log-likelihood for the inverse gamma distribution
    def nll(params, data, weights):
        shape, loc, scale = params
        pdf_values = stats.invgamma.pdf(data, shape, loc, scale)
        return -np.sum(weights * np.log(pdf_values))  # Weighted negative log-likelihood
     
    # Fit inverse gamma distribution to the binned data (using the bin centers and normalized histogram as weights)
    initial_params = [1, 0, 1]  # Initial guess for shape, location, and scale
    result1 = minimize(nll, initial_params, args=(bin_centers1, hist1_normalized), bounds=((0.001, None), (0, 0), (0.001, None)))
    result2 = minimize(nll, initial_params, args=(bin_centers2, hist2_normalized), bounds=((0.001, None), (0, 0), (0.001, None)))
     
    # Extract fitted parameters
    shape1, loc1, scale1 = result1.x
    shape2, loc2, scale2 = result2.x
     
    # Display fitted parameters
    print(f"Dataset 1 - Fitted shape: {shape1}, scale: {scale1}")
    print(f"Dataset 2 - Fitted shape: {shape2}, scale: {scale2}")
     
    # Create log-spaced points for continuous plotting of the PDF
    x_values = np.logspace(np.log10(min(bin_centers1.min(), bin_centers2.min())), np.log10(max(bin_centers1.max(), bin_centers2.max())), 500)
     
    # Calculate the continuous PDF values for the fitted distributions
    pdf1 = stats.invgamma.pdf(x_values, shape1, loc1, scale1)
    pdf2 = stats.invgamma.pdf(x_values, shape2, loc2, scale2)
     
    fig2 = plt.figure(figsize=(12,8), layout='constrained')
    # Plot the binned data and continuous fitted PDF
    plt.scatter(bin_centers1, hist1_normalized, color='0.2',  s=30, zorder=5)
    plt.scatter(bin_centers2, hist2_normalized, color='0.8',  s=30, zorder=5)
     
    plt.plot(x_values, pdf1, color='0.2', label='Fitted Inverse Gamma PDF - Roback Data', linewidth=2)
    plt.plot(x_values, pdf2, color='0.8', label='Fitted Inverse Gamma PDF - Model Data', linewidth=2)
     
    # Set axis scales to log-log
    plt.xscale('log')
    plt.yscale('log')
     
    # Adjust the x and y limits to exclude very small values (outliers)
    plt.xlim(1, max(x_values))
    plt.ylim(1e-11, max(max(pdf1), max(pdf2)))
     
    plt.legend()
    plt.show()
     
    # Perform the Kolmogorov-Smirnov test between the two datasets
    ks_stat, ks_p_value = ks_2samp(data1, data2)
     
    # Output the result of the KS test
    print(f"KS Statistic: {ks_stat}")
    print(f"KS p-value: {ks_p_value}")
     
    # Interpretation of the test result
    if ks_p_value > 0.05:
        print("The distributions are statistically similar (fail to reject the null hypothesis).")
    else:
        print("The distributions are statistically different (reject the null hypothesis).")
