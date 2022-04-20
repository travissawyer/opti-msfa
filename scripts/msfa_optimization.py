# Python scripts and associated functions for multispectral filter array (MSFA) optimization.
#
#
#
# Author: Travis Sawyer
#
# Date: May 31, 2020

# Import relevant packages.
import numpy as np
import os
from numpy import *
from scipy.signal import convolve2d
from scipy.interpolate import interp1d
import pandas as pd
from itertools import permutations, product
from pysptools import abundance_maps
from random import sample as randsamp
from scipy.io import loadmat, savemat
import random  # MTW
import math


### FILE HANDLING CODE START ###

def load_reference_data(data_filename):
    '''
    Loads a reference dataset from .mat format and associated endmembers / abundance map.
    '''
    d = loadmat(data_filename)
    em = loadmat(os.path.join(os.path.dirname(data_filename),'endmembers','endmembers.mat'))
    dinfo = pd.read_csv(os.path.join(os.path.dirname(data_filename),'info.csv'),
                        header=None,index_col=None,names=["parameter",'value','unit'])
    
    nrow = int(dinfo[dinfo['parameter'] == 'nrow'].value.values[0])
    ncol = int(dinfo[dinfo['parameter'] == 'ncol'].value.values[0])

    nbands = int(dinfo[dinfo['parameter'] == 'nbands'].value.values[0])
    spec_start = dinfo[dinfo['parameter'] == 'spec_start'].value.values[0]
    spec_end = dinfo[dinfo['parameter'] == 'spec_end'].value.values[0]
    
    data = d['Y']
    try:
        spec_bands = d['SlectBands']
    except:
        spec_bands = arange(0,nbands)
            
    # Define wavelength array
    wavelength = linspace(spec_start,spec_end,nbands)
    wavelength = wavelength[spec_bands].reshape(len(spec_bands))

    data = data / data.max()
    hypercube = data.reshape(len(wavelength), ncol, nrow).T # reshape hypercube

    if len(em['M']) > len(wavelength):
        endmembers = em['M'][spec_bands]
    else:
        endmembers = em['M']

    endmembers = endmembers.reshape(len(wavelength),-1)
    endmembers = c_[wavelength, endmembers].T
    
    a_map = em["A"].reshape((len(endmembers)-1, ncol, nrow)).T # reshape abundance map
    
    # TODO: Calculate noise as difference between hypercube and a_map
    noise = zeros_like(hypercube)
    try:
        filters = np.load(data_filename.replace(".mat","_filters.npy"))
        try:
            corr = np.load(data_filename.replace(".mat","_corr.npy"))
        except:
            print("Did not find correlation matrix data. Generating now.")
            filter_responses = generate_filter_data(filters, wavelength)
            corr = generate_correlation_matrix(hypercube, filter_responses, wavelength)
            np.save(data_filename.replace(".mat","_corr.npy"),corr)
    except:
        print("Did not find pre-existing filter data")
        filters = None
        corr = None
    try:
        mosaic = np.load(data_filename.replace(".mat","_mosaic.npy"))
    except:
        print("Did not find pre-existing mosaic data")
        mosaic = None
        
    return hypercube, a_map, endmembers, noise, filters, corr, mosaic

def load_target_data(data_filename):
    '''
    Loads the target data. Required to have abundance map and endmembers.
    Checks to see if noise, filters and correlation matrix exist.
    '''    

    data = np.load(data_filename)
    try:
        endmembers = np.load(data_filename.replace(".npy","_endmembers.npy"))
    except:
        print("Did not find endmember data. Please provide to continue.")
        return
    
    try:
        noise = np.load(data_filename.replace(".npy","_noise.npy"))
    except:
        noise = zeros_like(data)
    try:
        filters = np.load(data_filename.replace(".npy","_filters.npy"))
        try:
            corr = np.load(data_filename.replace(".npy","_corr.npy"))
        except:
            print("Did not find correlation matrix data. Generating now.")
            wavelength = endmembers[0,:]
            spectra = endmembers[1:,:]   
            hypercube, _ = create_synthetic_hypercube(data, spectra, wavelength)
            filter_responses = generate_filter_data(filters, wavelength)
            corr = generate_correlation_matrix(hypercube, filter_responses, wavelength)
            np.save(data_filename.replace(".npy","_corr.npy"),corr)
    except:
        print("Did not find pre-existing filter data")
        filters = None
        corr = None
    try:
        mosaic = np.load(data_filename.replace(".npy","_mosaic.npy"))
    except:
        print("Did not find pre-existing mosaic data")
        mosaic = None

    return data, endmembers, noise, filters, corr, mosaic


### HYPERCUBE GENERATION CODE START ###

def create_synthetic_hypercube(a_map, spectra, wavelength):
 
    '''
    Encodes a 2d image with spectral data to generate a synthetic hypercube.
    
    Inputs:
        a_map (N x M x L array) - 3D abundance map of target. Each channel in L corresponds to the
             abundance (between 0 and 1) of the spectral signature in the spectra array. 
             
        spectra (L x Q array) - Array of 1D spectral respones for each pixel. Number of array members
            should be equal to the maximum value in im (L). Q is sample points in wavelength space.
            
        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            response arrays.
    
    Output:
        hypercube (N X M x Q) - 3D synthetic hypercube.
        
        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            dimension of the hypercube.
    '''

    N,M,_ = a_map.shape
    L,Q = spectra.shape

    hypercube = zeros((N,M,Q))

    for i in range(L):
        hypercube += outer(a_map[:,:,i],spectra[i]).reshape((N,M,Q))

    return hypercube, wavelength

def generate_correlation_matrix(hypercube, filter_responses, wavelength, verbose=False):
    '''
    Function to generate the correlation matrix between different
    channels in the MSFA. Computes the correlation between the central 
    wavelengths of each filter for every unique spectral response
    found in the target. Then, takes a weighted average of these
    correlations depending on how prevalent the particular spectral
    response is. Correlation matrix should be calculated before
    noise is added to the system.
    
    Inputs:
        hypercube (N x M x Q) - 3D synthetic hypercube.
        
        filter_responses (J x Q) - Normalized (max of 1) spectral response curves, or QE curves, for each 
            filter in the MSFA. A total of J different filter responses of the macropixel
            should be provided in reading order. J will equal the number of sub-pixels in the macro-pixel.
            e.g., px*py = J. Even if there is a duplicate filter, provide the same spectral response curve.
            
        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            dimension of the hypercube. Filter spectral responses must also be provided on this axis.
        
        verbose (optional Boolean, default=False) - Set to true to get updates on processing
        
    Outputs:
        corr (J x J) - Correlation matrix between central wavelengths 
    '''
    
    # Normalize spectral data per pixel to 
    N,M,Q = hypercube.shape
    hsi_norm = hypercube.max(axis=2)
    hsi_norm[hsi_norm==0] = 1.

    responses = []
    num_responses = []

    for i in range(N):
        if verbose and not i%100:
            print('Processing Row %d' %i)
        for j in range(M):
            norm_sig = hypercube[i,j]/hsi_norm[i,j]
            index_array = [all(norm_sig == resp) for resp in responses]
            if any(index_array):
                num_responses[argmax(index_array)]+=1
                continue
            else:
                responses.append(norm_sig)
                num_responses.append(1)
                
    filter_maxima = [argmax(f) for f in filter_responses]

    m_sigs = []
    for r in responses:
        max_signal = r[filter_maxima]
        m_sigs.append(max_signal)
        
    covariance = cov(m_sigs, fweights = num_responses, rowvar=False)
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return abs(correlation)

def generate_filter_data(filter_info, wavelength):
    '''
    Converts data from saved filter information (band locations, bandwidths, peak transmissions)
    into gaussian filter responses over a wavelength array.
    
    Inputs
        filter_info (L x 3 array): Saved filter information array in the form of (cw, bws, trans)
            where cw is Center Wavlength, bws is bandwidths, and trans is peak transmission.
    
        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            response arrays.
            
    Outputs
        filter_data (L x Q array) - Array of filter responses over the wavelength array.
    '''
    L,_ = filter_info.shape
    Q = len(wavelength)
    
    filter_data = zeros((L,Q))
    
    for i in range(L):
        cw, bw, t = filter_info[i]
        sigma = bw / 2.355 # Convert from FWHM to sigma
        filter_data[i] = exp(-(wavelength - cw)**2/(2*sigma**2))*t
    
    return filter_data


### SPATIAL OPTIMIZATION CODE START ### 

def sample_hypercube_MSFA(hypercube, pattern_dims, filter_responses, wavelength, full_res=False):
    '''
    Generate raw MSFA data from a synthetic hypercube.
    
    Inputs:
        hypercube (N x M x Q) - 3D synthetic hypercube.
        
        pattern_dims ((my, mx)) - Tuple of dimensions for macropixel. A 3x3 pixel would be
            provided as (3,3) whereas a 3x2 pixel is provided as (3,2), etc.
        
        filter_responses (J x Q) - Normalized (max of 1) spectral response curves, or QE curves, for each 
            filter in the MSFA. A total of J different filter responses of the macropixel
            should be provided in reading order. J will equal the number of sub-pixels in the macro-pixel.
            e.g., px*py = J. Even if there is a duplicate filter, provide the same spectral response curve.
            
        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            dimension of the hypercube. Filter spectral responses must also be provided on this axis.
            
        full_res (optional Boolean) - Set to true to generate a reference image with no spatial sampling
            effects from the MSFA pattern pixellation. Use to generate "perfect" spatial image.
    
    Outputs:
        raw_MSFA (N x M x J) - Series of 2D images showing the collected signal for
            a given channel.
    '''
    my,mx = pattern_dims
    N,M,Q = hypercube.shape
    J = filter_responses.shape[0]
    
    if not mx*my == J:
        print('The filter response array does not have the correct number of members')
        return
    
    raw_MSFA = zeros((N,M,J))
    n = 0
    for i in range(mx):
        for j in range(my):
            
            MSFA = zeros((N,M))
            filter_response = filter_responses[n]
            
            
            if full_res:
                MSFA = sum(hypercube*filter_response,axis=2)
            else:
                MSFA[j::my,i::mx] = sum(hypercube[j::my,i::mx,:]*filter_response,axis=2)
            
            raw_MSFA[:,:,n] = MSFA
            
            n+=1

    return raw_MSFA, wavelength

def compute_H(my,mx):
    '''
    Caclulates the bilinear interpolation filter for a given mosaic.
    
    Inputs:
        mx -  Horizontal array dimension of MSFA macro-pixel.
        
        my - Vertical array dimension of MSFA macro-pixel.
        
    Outputs:
        H (mx*2-1 x my*2-1) - Weighted, normalized bilinear interpolation filter.
    '''

    # Initialize empty array
    H = zeros((int(my)*2-1,int(mx)*2-1))
    
    # Define un-normalized filter
    for i in range(my+1):
        H[i] = r_[arange(1,mx+1),arange(1,mx)[::-1]]*(i+1)

    for i in range(my):
        H[my*2-2-i] = r_[arange(1,mx+1),arange(1,mx)[::-1]]*(i+1)
        
    # Normalize
    H /= H.max()
        
    return H

def WB(I,H):
    '''
    Weighted bilinear interpolation.
    
    Inputs: 
        I (N x M array) - Individual channel image from MSFA
        
        H (2D array) - Weighted, normalized bilinear interpolation filter. Calculated from 
            mosaic dimensions
        
    Outputs:
        I_wb - Interpolated channel image.
        
    '''
    I_wb = convolve2d(I,H,mode='same')
    return I_wb

def binary_mask_MSFA(N,M,mx,my,L):
    '''
    Function to create binary mask corresponding to "observed" pixels
    in a MSFA channel.
    
    Inputs: 
        N,M - MSFA image data dimensions
        
        mx -  Horizontal array dimension of MSFA macro-pixel.
        
        my - Vertical array dimension of MSFA macro-pixel.
        
        L - Filter number (reading order)
    
    Outputs:
        mask (N,M) - Binary mask for observed pixels of a given channel.
    '''
    # This is not very elegant...
    output = zeros((N,M))
    
    n = 0
    for i in range(mx):
        for j in range(my):
            if n == L:
                output[j::my,i::mx] = 1.
                return output
            else:
                n+=1
                
    return

# Iterative spectral differences
def ISB(I, pattern_dims, corr, iteration=0,verbose=False):
    '''
    Iterative spectral differences algorithm for demosaicking.
    
    Inputs:
        I (N x M x J array) - MSFA raw data array with J different channels. 
        
        pattern_dims ((my, mx)) - Tuple of dimensions for macropixel. A 3x3 pixel would be
                provided as (3,3) whereas a 3x2 pixel is provided as (3,2), etc.
        
        corr (J x J array) - Correlation matrix describing spectral correlation
            between each filter. Used to determine maximum number of iterations
        
        iteration (int, default=0) - Iteration number, tracked outside of function.
        
        verbose (optional Boolean) - Set to true for status updates. 
        
    Outputs:
        demosaicked (N x M x L array) - Demosaicked MSFA data
    
    '''
    if verbose and not iteration % 10:
        print("Iteration %d" % iteration)
    
    N,M,channels = I.shape
    my,mx = pattern_dims
    
    Nab = exp(corr*3.) # Compute iterations per channel pair
    
    demosaicked = zeros_like(I)
    
    H = compute_H(my,mx)
    
    # Reference - ch1
    for ch1 in range(channels):
        B1 = binary_mask_MSFA(N,M,mx,my,ch1)
        
        # Set output data to observed 
        demosaicked[:,:,ch1][B1==1.] = I[:,:,ch1][B1==1.]
        # Target - ch2
        for ch2 in range(channels):
            if ch1==ch2 or Nab[ch1,ch2]<=iteration:
                continue
            else:
                B2 = binary_mask_MSFA(N,M,mx,my,ch2)
                
                # Demosaicked channel A
                if iteration==0:
                    Ca = WB(I[:,:,ch1],H)
                else:
                    Ca = I[:,:,ch1]
                # Apply binary mask for target B
                # Subtract from observed to get difference
                # Apply bilinear interpolation
                Kab = WB(Ca*B2 - I[:,:,ch2]*B2,H)
                
                # 
                demosaicked[:,:,ch2][B1==1.] = (Ca - Kab)[B1==1.]

    return demosaicked

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))

class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    TODO: Implement gamma correction

    """
    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    cmf = np.loadtxt('cie-cmf.txt', usecols=(1,2,3))
    wv = arange(380,785,5)
                         
    def __init__(self, red, green, blue, white, wavelength = None):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.
        
        If wavelength is provided, then the CMF function will be re-scaled
        to be on the same sampling as the provided array.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]
        if not wavelength is None:
            self.cmf = array([interp1d(self.wv,self.cmf[:,0], fill_value="extrapolate")(wavelength),
                              interp1d(self.wv,self.cmf[:,1], fill_value="extrapolate")(wavelength),
                              interp1d(self.wv,self.cmf[:,2], fill_value="extrapolate")(wavelength)]).T
            self.wv = wavelength

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point.

        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """

        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum to an rgb value."""

        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)
    
def MSFA_to_RGB(MSFA, filters, wavelength, cs, verbose=False, alpha=False):
    '''
    Convert demosaicked MSFA data to RGB spectra using filter responses to
    represent complete spectra.
    
    Inputs:
    
        MSFA (N x M x J) - MSFA demosaiced data array with J channels / filters. 
        
        filters (J x Q) - Normalized (max of 1) spectral response curves, or QE curves, for each 
            filter in the MSFA. A total of J different filter responses of the macropixel
            should be provided in reading order. J will equal the number of sub-pixels in the macro-pixel.
            e.g., px*py = J. Even if there is a duplicate filter, provide the same spectral response curve.
        
        wavelength (Q x 1) - Array describing the wavelength value corresponding to the spectral
            dimension of the hypercube. Filter spectral responses must also be provided on this axis.
        
        cs - ColorSystem object defining RGB colorspace
        
        verbose (optional) - Set to true for status updates.
        
        alpha (optional) - Return data as RGBA where A is the alpha channel.
    
    Outputs: 
        MSFA_RGB (N x M x 3 array) - RGBA-equivalent image generated from MSFA data
    
    '''
    px,py,channels = MSFA.shape
    
    MSFA_RGB = zeros((px,py,3))
    
    if alpha:
        alpha_ch = zeros((px,py))
    
    # Can we do this with an array operation to speed up the 
    # processing?
    
    for j in range(px):
        if verbose and not j % 100:
            print("Converting row %d" %j)
        for i in range(py):
            spectra = zeros_like(filters[0])
            for c in range(channels):
                spectra += MSFA[j,i,c]*filters[c]
            MSFA_RGB[j,i,:] = cs.spec_to_rgb(spectra)

    if alpha:
        alpha_ch = MSFA.max(axis=2) / MSFA_RGB.max()
        return MSFA_RGB, alpha_ch
    else:
        return MSFA_RGB


def hypercube_to_RGB(hypercube,wavelength,cs,verbose=False, alpha=False):
    '''
    Convert hyperspectral data to RGB
    
    Inputs:
        hypercube (N X M x Q) - 3D synthetic hypercube.
        
        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            response arrays.    
            
        cs - ColorSystem object defining RGB colorspace
        
        verbose (optional) - Set to true for status updates.
        
        alpha (optional) - Return data as RGBA where A is the alpha channel.
        
    Outputs:
        hypercube_RGB (N x M x 3) - RGB-equivalent image generated from hyperspectral data.
        
        alpha_ch (N,M optional) - Transparency mask for pixel brightness
    
    '''
    px,py,wv = hypercube.shape
    
    hypercube_RGB = zeros((px,py,3)) # RGB 
    if alpha:
        alpha_ch = zeros((px,py))
    
    # Can we do this with an array operation to speed up the 
    # processing?

    for j in range(px):
        if verbose and not j % 100:
            print("Converting row %d" %j)
        for i in range(py):
            hypercube_RGB[j,i,:3] = cs.spec_to_rgb(hypercube[j,i])

    if alpha:
        alpha_ch = hypercube.max(axis=2) / hypercube.max()
        return hypercube_RGB, alpha_ch
    else:
        return hypercube_RGB

def RMS_pixel_difference(im,ref):
    '''
    Total RMS error between an image (im) and reference (ref). Used as a merit function.
    
    Inputs:
        im (N x M array) - Image to test quality.
    
        ref (M x M array) - Reference to compare test image to.
    
    Outputs:
    
        rms (float): RMS pixel difference between input image and a reference.
    
    '''
    
    # How to account for scaling between channels?
    d = im / im.max(axis=(0,1))
    rms = sqrt( mean((d-ref)**2) )
        
    return rms

def DFT_difference(im,ref):
    '''
    Merit function option to compute difference between two images
    in fourier space using discrete fourier transform.
    
        Inputs:
        im (N x M array) - Image to test quality.
    
        ref (M x M array) - Reference to compare test image to.
    
    Outputs:
    
        dft_diff (float): 
    
    
    '''
    # TO DO: Write this function
    
    return dft_diff

def demosaick_input_image(hypercube,wavelength,pattern_dims, filters, corr,demosaicking='ISD'):
    raw, _ = sample_hypercube_MSFA(hypercube,pattern_dims,filters, wavelength)

    if demosaicking == "WB":
        H = compute_H(*pattern_dims)
        demosaicked = zeros_like(raw)
        for c in range(len(filters)):
            demosaicked[:,:,c] = WB(raw[:,:,c],H)
    elif demosaicking == "SD":
        demosaicked = ISB(raw,pattern_dims, corr, iteration=0,verbose=verbose)
    else:
        for iteration in range(int(exp(corr.max()*3.))):
            demosaicked = ISB(raw,pattern_dims, corr, iteration=iteration,verbose=verbose)  
    return demosaicked

def all_filters_present(avail_fils,N_filters):
    '''
    Helper function to make sure that only mosaic patterns including all available filters are used. This comes into play when using fewer filters than mosaic pattern locations.
    '''
    combos = []
    for total in product(avail_fils, repeat=N_filters):
        valid = 1
        for i in avail_fils:
            if not i in total:
                valid = 0
        if valid == 1:
            combos.append(total)
        
    return combos
        
def spatial_optimization_MSFA(hypercube,wavelength,pattern_dims, filters, corr, filter_labels=None, demosaicking='ISD',op_type='exhaustive', merit_function='RMS',verbose=False):
        '''
        Full spatial optimization pipeline to determine the optimal filter
        pattern for a multispectral filter array given a synthetic hypercube
        target, and filter responses.
        
        Inputs:
            hypercube (N X M x Q) - 3D synthetic hypercube.
            
            wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
                response arrays.
            
            pattern_dims ((my, mx)) - Tuple of dimensions for macropixel. A 3x3 pixel would be
                provided as (3,3) whereas a 3x2 pixel is provided as (3,2), etc.

            filters (J x Q) - Normalized (max of 1) spectral response curves, or QE curves, for each 
                filter in the MSFA. A total of J different filter responses of the macropixel
                should be provided in reading order. J will equal the number of sub-pixels in the macro-pixel.
                e.g., px*py = J. Even if there is a duplicate filter, provide the same spectral response curve.

            filter_labels (optional J x 1 List): List of labels for each filter
            
            demosaicking (optional): 
                'ISD' - Iterative spectral differences (default) 
                'SD' - Spectral differences
                'WB' - Weighted bi-linear interpolation
        
            op_type (optional): Type of optimization.
                'exhaustive' - test every possible spatial arrangement
        
            merit_function (optional):
                'RMS' - Computes RMS pixel differences between (default)
                'RMS-RGB' - Computes RMS pixel differences between
                'DFT'
                'DFT-RGB'
            
            verbose (optional): Set to true for status updates
            
        Outputs:
            pattern
        '''
        # Total number of filters in the pattern
        N_filters = pattern_dims[1]*pattern_dims[0]#len(filters)
        # Available filters to place
        avail_fils = arange(len(filters))
        
        # Iterate through every possible permutation of the filter arrangement
        if op_type == 'exhaustive':
            best = inf
            pattern = []
            best_demosaicked = None
            results = []
            print("Running exhaustive spatial optimization")

            filter_combos = all_filters_present(avail_fils,N_filters)
            for fil_set in filter_combos:

                
                # Re-organize filters appropriately
                f = array([filters[i] for i in fil_set])
                
                # Generate raw MSFA data
                raw, _ = sample_hypercube_MSFA(hypercube,pattern_dims,f,wavelength)

                # Generate perfect spatial reference
                ref,_ = sample_hypercube_MSFA(hypercube,pattern_dims,f,wavelength,full_res=True)
   
                # Demosaic the raw MSFA data using iterative spectral differences
                if demosaicking == "WB":
                    #if verbose:
                        #print("Using weighted bilinear interpolation for demosaicking.")
                    H = compute_H(*pattern_dims)
                    demosaicked = zeros_like(raw)
                    for c in range(len(f)):
                        demosaicked[:,:,c] = WB(raw[:,:,c],H)
                elif demosaicking == "SD":
                    #if verbose:
                        #print("Using spectral differences for demosaicking")
                    demosaicked = ISB(raw,pattern_dims, corr, iteration=0,verbose=verbose)
                else:
                    #if verbose:
                        #print("Using iterative spectral differences with %i iterations" %int(exp(corr.max()*3.)))
                    for iteration in range(int(exp(corr.max()*3.))):
                        demosaicked = ISB(raw,pattern_dims, corr, iteration=iteration,verbose=verbose)                
                if merit_function == 'RMS-RGB':
                    illuminant_D65 = xyz_from_xy(0.3127, 0.3291)

                    cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                                               green=xyz_from_xy(0.30, 0.60),
                                               blue=xyz_from_xy(0.15, 0.06),
                                               white=illuminant_D65)

                    demosaicked_RGB = MSFA_to_RGB(demosaicked, filters, wavelength,cs)
                    ref_RGB = hypercube_to_RGB(hypercube,wavelength,cs)

                    m = RMS_pixel_difference(demosaicked_RGB, ref_RGB)
                else: # Default
                    m = RMS_pixel_difference(demosaicked, ref)

                results.append(m)

                if m < best:
                    best = m
                    pattern = fil_set
                    best_demosaicked = demosaicked

                if verbose and filter_labels:
                    print("Using pattern (reading order) of ", [filter_labels[i] for i in fil_set], "gives merit function result of", results[-1])
        else:
            print("Optimizations other than exhaustive are not yet supported.")
            return False
        print("Optimal pattern is", pattern)
        return pattern, demosaicked
    
########### SPECTRAL BAND OPTIMIZATION CODE START 


def compute_unmixing_accuracy(hypercube, abundance_map, endmembers, wavelength, center_wavelengths, bandwidths, transmission=None, return_predicted=False, filtered_input=False):
    '''
    Function to compute unmixing accuracy using NNLS spectral unmixing given a set of spectral band 
    center wavelengths and bandwidths. Assuming gaussian filter responses with a bandwidth defined
    as the FWHM.
    
    Inputs:
        hypercube (N X M x Q) - 3D synthetic hypercube.
        
        abundance_map (N x M x P)- 3D ground-truth abundance map of target. Each channel in L corresponds to the
             abundance (between 0 and 1) of the spectral signature in the spectra array. 
        
        endmembers (P x Q array) - Array of endmember signals to unmix.
        
        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            response arrays.
        
        center_wavelengths (1 x L array) - Array or list of center wavelengths for Gaussian filter responses
            to use for unmixing.
        
        bandwidths (1 x L array) - Array or list of bandwidths for Gaussian filter responses
            to use for unmixing.
            
        transmission (optional)
        
        return_predicted (optional)
        
        filtered_input (optional)
        
    Outputs:
        accuracy (float) - RMS error of the unmixed abundance compared to the gruond truth
        
        predicted (optional N x M x P array) - Predicted abundance map of hypercube
    '''
    # Extract relevant dimensions
    N,M,Q = hypercube.shape
    
    # Define filter responses
    if transmission:
        fil = c_[center_wavelengths, bandwidths, transmission]
        filters = generate_filter_data(fil, wavelength)
    else:
        fil = c_[center_wavelengths, bandwidths, ones_like(center_wavelengths)]
        filters = generate_filter_data(fil, wavelength)
    
    P = len(endmembers)
    L = len(filters)
    
    # Pre-allocate arrays
    data_sig = zeros((N*M,L))
    endmember_sig = zeros((P,L))
    
    # Iterate through each filter
    for i,f in enumerate(filters):
        # Compute the signal from each filter received from the data
        # and that received from an endmember.
        if filtered_input: # Already have filters applied
            data_sig[:,i] = sum(hypercube.reshape((N*M,Q)),axis=1) # (N*M) x L
        else:
            data_sig[:,i] = sum(hypercube.reshape((N*M,Q))*f,axis=1) # (N*M) x L
        
        for j, e in enumerate(endmembers):
            endmember_sig[j,i] = sum(endmembers[j]*f) # P x Q
            
    # Reshape the data array into 1d array
    predicted = abundance_maps.amaps.NNLS(data_sig, endmember_sig) # (N*M x P)

    # Take RMS difference between true abundance map and computed.
    accuracy = sqrt(mean((abundance_map.reshape((N*M,len(endmember_sig)))-predicted)**2))
    
    if not accuracy or isnan(accuracy):
        return 100.
    
    if return_predicted:
        return accuracy, predicted.reshape((N,M,P))
    else:
        return accuracy
    
def increment_position(hypercube,abundance_map, endmembers, wavelength, score, inc, pmax, pmin, score_function, bands, bws, i, j, verbose=False):
    '''
    Helper function to optimize the central wavelengths of spectral bands. Adjusts the center wavelength in one direction
    and compares to the previous result.
    '''
    if bands[i,j] >= pmax:
        bands[i,j] = pmax
        if verbose:
            print("Band location is at the maximum limit.")
        return score, bands, bws
    elif bands[i,j] <= pmin:
        bands[i,j] = pmin
        if verbose:
            print("Band location is at the minimum limit.")
        return score, bands, bws
    else:
        bands[i,j] += inc
        new_score = score_function(hypercube,abundance_map, endmembers, wavelength, bands[i], bws[i])
        if new_score < score:
            if verbose:
                print("Position adjustment resulted in previous score of %.3g improving to %.3g." %(score, new_score))
            return increment_position(hypercube,abundance_map, endmembers, wavelength, new_score, inc, pmax, pmin, score_function, bands, bws, i, j,verbose=verbose)
        else:
            if verbose:
                if inc < 0:
                    print("Convergence reached for band position adjustment in negative direction.")
                else:
                    print("Convergence reached for band position adjustment in positive direction.")
            bands[i,j] -= inc
            return score, bands, bws
        
def increment_bandwidth(hypercube,abundance_map, endmembers, wavelength, score, inc, bmax, bmin, score_function, bands, bws, i, j,verbose=False):
    '''
    Helper function to optimize the bandwidths of spectral bands. Adjusts the bandwidth in one direction
    and compares to the previous result.
    '''
    if bws[i,j] >= bmax:
        bws[i,j] = bmax
        if verbose:
            print("Bandwidth is at the maximum limit.")
        return score, bands, bws
    elif bws[i,j] <= bmin:
        bws[i,j] = bmin
        if verbose:
            print("Bandwidth is at the minimum limit.")
        return score, bands, bws
    else:
        bws[i,j] += inc
        new_score = score_function(hypercube,abundance_map, endmembers, wavelength, bands[i], bws[i])
        if new_score < score:
            if verbose:
                print("Bandwidth adjustment resulted in previous score of %.3g improving to %.3g." %(score, new_score))
            return increment_bandwidth(hypercube,abundance_map, endmembers, wavelength, new_score, inc, bmax, bmin, score_function, bands, bws, i, j,verbose=verbose)
        else:
            if verbose:
                if inc < 0:
                    print("Convergence reached for bandwidth adjustment in negative direction.")
                else:
                    print("Convergence reached for bandwidth adjustment in positive direction.")
            bws[i,j] -= inc
            return score, bands, bws

def refine_position(hypercube,abundance_map,endmembers,wavelength, bands, bws, maxiter=20,score_function=compute_unmixing_accuracy,pstep=2,verbose=False):
    '''
    Helper function to optimize the central wavelengths of spectral bands. This function effectively conducts a gradient
    descent optimization to adjust the center wavelength until a local minimum is reached.
    '''
    old_ps = zeros_like(bands) # Empty array to track changes
    n = 0 # index of iterations

    # First, iterate through positions
    while any(old_ps != bands) and n < maxiter:
        old_ps = bands.copy()
        n += 1
        if verbose:
            print("Performing position refinement iteration number %d." %n)
                    
        for i, band in enumerate(bands):
            score = score_function(hypercube,abundance_map, endmembers, wavelength, band, bws[i])

            for j, p in enumerate(band):
                if verbose:
                    print("Refining position of band %.1f nm. Beginning score of %.3g." % (p,score))
                score, bands, bws = increment_position(hypercube,abundance_map, endmembers, wavelength,score, pstep, int(wavelength[-1]-5), int(wavelength[0]+5), score_function, bands, bws, i ,j,verbose=verbose)
                score, bands, bws = increment_position(hypercube,abundance_map, endmembers, wavelength, score, -1*pstep, int(wavelength[-1]-5), int(wavelength[0]+5), score_function, bands, bws, i, j,verbose=verbose)

    if verbose:
        print("Final band locations for this loop are ",bands)

    return bands, bws

        
def refine_bandwidth(hypercube,abundance_map,endmembers,wavelength, bands, bws , maxiter=20,score_function=compute_unmixing_accuracy,bstep=2,bw_max=50.,bw_min=20.,verbose=False):              
    '''
    Helper function to optimize the bandwidth of spectral bands. This function effectively conducts a gradient
    descent optimization to adjust the bandwidth until a local minimum is reached.
    '''
    old_bws = zeros_like(bws) # empty array to track changes
    n = 0 # index of iterations
            
    # Now, through bandwidths
    while any(old_bws != bws) and n < maxiter:
        old_bws = bws.copy()
        n += 1
        if verbose:
            print("Performing bandwidth refinement iteration number %d." %n)
        for i, band in enumerate(bands):
                   
            score = score_function(hypercube,abundance_map, endmembers, wavelength, band, bws[i])
            for j, z in enumerate(band):
                if verbose:
                    print("Refining bandwidth of band %.1f nm. Beginning score of %.3g." % (z, score))
                score, bands, bws  = increment_bandwidth(hypercube,abundance_map, endmembers, wavelength, score, bstep, bw_max, bw_min, score_function, bands, bws, i, j,verbose=verbose)
                score, bands, bws  = increment_bandwidth(hypercube,abundance_map, endmembers, wavelength, score, -1*bstep, bw_max, bw_min, score_function, bands, bws, i, j,verbose=verbose) # lazy way to check both directions
                
    if verbose:
        print("Final bandwidths for this loop are ",bws)
    return bands, bws

def optimize_spectral_bands(hypercube, abundance_map, endmembers, wavelength, N_filters, op_type_spectral, **kwargs):
    '''
    Wrapper function to delegate which optimization should be used for spectral bands.
    
    '''
    if op_type_spectral == "gradient-descent":
        bands,bws = optimize_spectral_gradient_descent(hypercube,abundance_map, endmembers, wavelength, N_filters, **kwargs)
    elif op_type_spectral == "simulated-annealing":
        bands,bws = optimize_spectral_simulated_annealing(hypercube,abundance_map, endmembers, wavelength, N_filters, **kwargs)
    else: # Default gardient descent
        bands,bws = optimize_spectral_gradient_descent(hypercube,abundance_map, endmembers, wavelength, N_filters, **kwargs)
    return bands, bws


def optimize_spectral_simulated_annealing(hypercube, abundance_map, endmembers, wavelength, N_filters,  bw_max=50., bw_min=20., bstep=2., wv_max=None, wv_min=None,pstep=2.,random_sampling=False, n_iter=10, loops=1,merit_function="unmixing",verbose=False):
    '''
    Inputs:
        hypercube (N X M x Q array) - 3D synthetic hypercube.

        abundance_map (N x M x P array)- 3D abundance map of target. Each channel in L corresponds to the
             abundance (between 0 and 1) of the spectral signature in the spectra array. 

        endmembers (P x Q array) - Array of endmember signals to unmix.

        wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
            response arrays.
                
        N_filters (int): Maximum number of spectral bands. (currently fixed)    
        
        n_iter(optional float, default=10): Cooling parameter = 10 / n_iter
        
        loops (optional float, default=1): Max temperature = loops*600
                       
        bw_max (optional float, default=50): Maximum allowable bandwidth in nanometers for a spectral band.

        bw_min (optional float, default=5): Minimum allowable bandwidth in nanometers for a spectral band.
            
        wv_max (optional float, default=match wavelength array): Maximum allowable central wavelength.
            
        wv_min (optional float, default=match wavelength array): Minimum allowable central wavelength.        
        
        merit_function (optional string, default="unmixing"): Merit function to use in
                the optimization. The options are:
                   'unmixing': Spectral unmixing accuracy. Minimizes the root-sum-squared difference
                       between the true endmember abundance and that obtained by NNLS unmixing.

        verbose (optional boolean, default=False): Print out relevant information
              as optimization proceeds. TODO: Set verbosity levels of 0-4 for different amounts
              of output information.
              
        Returns:
            center_wavelengths (1 x N_filters array) - Array describing center wavelengths of optimal
                filter set. 
            
            bandwidths (1 x N_filters array) - Array describing bandwidths of optimal
                filter set.           
    '''
          
    ii=1

    bands_final=[]
    bws_final= []

    while ii <6: 

        bw_init = (bw_max + bw_min)/2.

        # If wavelength bounds are not defined, make them equal to array bounds
        if wv_max is None:
            wv_max = int(wavelength[-1])
        if wv_min is None:
            wv_min = int(int(wavelength[0]))    

        # TODO: Implement other merit functions
        if merit_function == "unmixing":
            score_function = compute_unmixing_accuracy
        else:
            score_function = compute_unmixing_accuracy

        # Start with a random sampling of spectral bands in the available range
        sbands = randsamp(range(int(wv_min-bw_init), int(wv_max-bw_init), int(bw_init*2)), N_filters)

        # Merit function for initial system state
        dist = score_function(hypercube,abundance_map, endmembers, wavelength, sbands, ones(N_filters)*bw_init)

        T_initial = 600.*loops # Initial temperature
        T_final = 2 # Final temperature
        a = 10/n_iter # Cooling parmaeter

        T=T_initial # Set inital Temp and Bandwidth to intial values

        bands=sbands  # Set initial bands and bandwidths to intial values
        bw=ones(N_filters)*bw_init

        # Start simulated annealing here to change bandwidths and band locations

        while T>T_final:

            M = score_function(hypercube,abundance_map, endmembers, wavelength, bands, bw)*sum(bw)/(N_filters*bw_init)

            change_bands=int(random.choice(range(1,N_filters))) # Changes a random band a random amount between -10 and 10
            bands_new=bands
            bands_new[change_bands]=( bands[change_bands] + random.choice(range(-10,10)))                    
            if bands_new[change_bands]<wv_min: 
                bands_new[change_bands]=wv_min+random.choice(range(-10,10))
            elif bands_new[change_bands]>wv_max: 
                bands_new[change_bands]=wv_max-random.choice(range(-10,10))


            change_bw=int(random.choice(range(1,N_filters)))  # Changes a random bandwidht a random amount between -10 and 10 unless the bwndwidth is less than the minimum
            bw_new=bw
            bw_new[change_bw]=( bw[change_bw] + random.choice(range(-10,10)) )

            if bw_new[change_bw]<bw_min: 
                bw_new[change_bw]=bw_min
            elif bw_new[change_bw]>bw_max: 
                bw_new[change_bw]=bw_max

            M_new=score_function(hypercube, abundance_map, endmembers, wavelength, bands_new, bw_new)*sum(bw_new)/(N_filters*bw_init)

            bands_old=bands
            bw_old=bw 

            print (T, M-M_new,bands)

            if abs(M-M_new)>100000:
                T=T_initial # Initial temperature
                T_final = 2 # Final temperature
                a = 10/n_iter # Cooling parmaeter
                T=T_initial # Set inital Temp and Bandwidth to intial values
                sbands = randsamp(range(int(wv_min-bw_init), int(wv_max-bw_init), int(bw_init*2)), N_filters)
                bands=sbands  # Set initial bands and bandwidths to intial values
                bw=ones(N_filters)*bw_init
                print('Reset')
            elif M_new>M:          
                bands=bands_new
                bw=bw_new
            elif random.uniform(0, 1) < math.exp((M-M_new)/ T):
                bands=bands_new
                bw=bw_new   
                T=T-a
            else:
                bands=bands
                bw=bw

        bands_final.append(bands_old)
        bws_final.append(bw_old)

        print(bands_old)
        print(bw_old)

        ii=ii+1
        
    return bands_final, bws_final

          
def optimize_spectral_gradient_descent(hypercube, abundance_map, endmembers, wavelength, N_filters, bw_max=50., bw_min=20., bstep=2., wv_max=None, wv_min=None, pstep=2.,random_sampling=False, n_iter=1000, loops=10,merit_function="unmixing",verbose=False):
        '''
        Optimize the spectral band position and bandwidth. Begins by stochastically sampling the 
        space of possible options. The top 5 results are then retained and a gradient descent
        approach is taken to convergene on the optimal result. The function returns the best 
        result, along with the merit function score.

        Inputs:
            hypercube (N X M x Q array) - 3D synthetic hypercube.

            abundance_map (N x M x P array)- 3D abundance map of target. Each channel in L corresponds to the
                 abundance (between 0 and 1) of the spectral signature in the spectra array. 

            endmembers (P x Q array) - Array of endmember signals to unmix.

            wavelength (1 x Q array) - Array describing the wavelength value corresponding to the spectral
                response arrays.
                
            N_filters (int): Maximum number of spectral bands. (currently fixed)

            bw_max (optional float, default=50): Maximum allowable bandwidth in nanometers for a spectral band.

            bw_min (optional float, default=5): Minimum allowable bandwidth in nanometers for a spectral band.
            
            wv_max (optional float, default=match wavelength array): Maximum allowable central wavelength.
            
            wv_min (optional float, default=match wavelength array): Minimum allowable central wavelength.

            bstep (optional float, default=2): Step-size in nanometers to take when optimzing spectral bands.

            random_sampling (optional Boolean, default=False)

            n_iter (optional int, default=10000): Number of iterations for the stochastic sampling step.
            
            loops (optinal int, default=10): Number of gradient descent optimization loops.

            merit_function (optional string, default="unmixing"): Merit function to use in
                the optimization. The options are:
                   'unmixing': Spectral unmixing accuracy. Minimizes the root-sum-squared difference
                       between the true endmember abundance and that obtained by NNLS unmixing.

            verbose (optional boolean, default=False): Print out relevant information
              as optimization proceeds. TODO: Set verbosity levels of 0-4 for different amounts
              of output information.

        Returns:
            center_wavelengths (R x N_filters array) - Array describing center wavelengths of optimal
                filter set. R is either 5 or 1 depending on which stochastic sampling.
            
            bandwidths (R x N_filters array) - Array describing bandwidths of optimal
                filter set. R is either 5 or 1 depending on which stochastic sampling.
        '''
        # Initial bandwidth will be halfway between max and min
        bw_init = (bw_max + bw_min)/2.
        
        # Generate empty arrays for stochastic optimization step.
        band_list = zeros((n_iter,N_filters))
        dist_list = zeros(n_iter)
        
        if wv_max is None:
            wv_max = int(wavelength[-1])
        if wv_min is None:
            wv_min = int(int(wavelength[0]))
        # TODO: Implement other merit functions
        if merit_function == "unmixing":
            score_function = compute_unmixing_accuracy
        else:
            score_function = compute_unmixing_accuracy
            
        if random_sampling == True:
            if verbose:
                print("Beginning stochastic optimization of band position with %d iterations." %n_iter)
            for i in range(n_iter):
                if verbose and not i % 100:
                    print("Stochastic optimization iteration %d." %i)
                # Generate the random sampling of spectral bands over wavelegnth range

                sbands = randsamp(range(int(wv_min+bw_min/2), int(wv_max-bw_min/2), int(bw_min)), N_filters)

                dist = score_function(hypercube,abundance_map, endmembers, wavelength, sbands, ones(N_filters)*bw_init)
                dist_list[i] = dist
                band_list[i,:] = sbands

            # Sort according to accuracy
            dist_sorted = dist_list[dist_list.argsort()]
            bands_sorted = band_list[dist_list.argsort()]

            bands = bands_sorted[:5] # take top 5
            bws = ones_like(bands)*bw_init
            if verbose:
                print("Stochastic optimization complete. Initializing gradient descent with top 5 candidates.")
                print(bands)
        else:
            # Evenly space the array
            if verbose:
                print("Initializing with evenly spaced band array.")
            bands = linspace(wv_min+bw_init*2,wv_max-bw_init*2,N_filters).reshape(1,N_filters)
            bws = ones_like(bands)*bw_init
                
        if verbose:
            print("Beginning gradient-based optimization of bandwidth.")
            
        for k in range(loops):
            if verbose:
                print("Performing iteration loop %d of %d." %(k, loops))
            bands, bws = refine_position(hypercube,abundance_map,endmembers,wavelength,bands, bws, score_function=score_function, pstep=pstep,verbose=verbose)
            bands, bws = refine_bandwidth(hypercube,abundance_map,endmembers,wavelength,bands, bws,score_function=score_function, bw_max=bw_max, bw_min=bw_min, bstep=bstep,verbose=verbose)

        # Sort the bands in ascending order
        for i in range(len(bands)):
            slist = argsort(bands[i,:])
            bands[i,:] = bands[i,:][slist]
            bws[i,:] = bws[i,:][slist]
        
        # TODO: Output the filter info array directly (bands, bws, transmission)
        return bands, bws
                         
                         
# Define default colorspace                         
illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)                         
