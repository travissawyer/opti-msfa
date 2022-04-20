# Import relevant GUI packages
import tkinter as tk
import os, pickle
from tkinter import ttk, Canvas, PhotoImage, filedialog
from PIL import Image, ImageTk
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# Add scripts folder to import path
import sys
sys.path.insert(1, '../scripts')

# Import plotting packages
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import MSFA Optimization Code
import msfa_optimization
from msfa_optimization import *

# Generate GUI window
root = tk.Tk() 
root.title("Multispectral Filter Array Optimizer") 
root.geometry("1350x700")

tabControl = ttk.Notebook(root) 
  
tab1 = ttk.Frame(tabControl) 
tab2 = ttk.Frame(tabControl) 
tab3 = ttk.Frame(tabControl) 

# Define global variables
fname = "../input/USAF.png"
a_map = None
h_ref = None
endmembers = None
spectra = None
wavelength = None
noise = None
corr = None
filters = None
mosaic = None
bands = None
bws = None

# Define containers and default values for optimization variables
# Spectral optimization
N_bands  = 4
random_sampling = True
bw_max = 50
bw_min = 20
wv_max = None
wv_min = None
bstep = 2
pstep = 2
n_iter = 1000
loops = 10
merit_function = "unmixing"
op_type_spectral = "gradient-descent"

# Spatial Optimization
pattern_dims = (2,2)
demosaicking = "WB"
op_type = "exhaustive"
merit_function_spatial="RMS"

# Define default colorspace                         
illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)                         

img = Image.open(fname)  # PIL solution
img = img.resize((200, 200), Image.ANTIALIAS) #The (250, 250) is (height, width)
img = ImageTk.PhotoImage(img) # convert to PhotoImage
  
placeholder = img
    
tabControl.add(tab1, text ='MSFA Optimizer') 
tabControl.add(tab2, text ='Spectral Band Options') 
tabControl.add(tab3, text ='Spatial Mosaic Options') 
    
# Tab 1: Main window for optimization
# Call grid on separate line to avoid NoneType definition
fname_label = tk.Label(tab1, text = "Data currently selected: " + os.path.split(fname)[-1], width=50, anchor="w")
fname_label.grid(column = 1, row = 0, padx = 0, pady = 20) 

# Note that I will eventually have to change these to canvases since we'll be plotting
# data, not just displaying images.
disp_image_label = tk.Label(tab1, text="RGB-Equivalent of HSI Image", anchor='n')
disp_image_label.grid(column = 0, row = 2, padx = 20, pady = 0) 
disp_image = tk.Label(tab1,image=img, width=200, height=200)
disp_image.grid(column = 0, row = 1, padx = 20, pady = 0) 

demosaicked_label = tk.Label(tab1, text="Demosaicked Image", anchor='n')
demosaicked_label.grid(column = 0, row = 4, padx = 20, pady = 0) 
demosaicked_image = tk.Label(tab1 ,image=img, width=200, height=200,anchor='s')
demosaicked_image.grid(column = 0, row = 3, padx = 20, pady = (20,0)) 

# Create plot for endmembers
endmember_label = tk.Label(tab1, text="Endmembers / Optimal Bands", anchor='n')
endmember_label.grid(column = 2, row = 2, padx = 20, pady = 0) 
em_fig = Figure(figsize=(3,2))
a = em_fig.add_subplot(111)
a.set_ylabel("Normalized Signal", fontsize=8)
a.set_xlabel("Wavelength (nm)", fontsize=8)
a.set_ylim(-0.1,1.1)
a.set_xticklabels(a.get_xticklabels(),fontsize=8)
a.set_yticklabels(a.get_yticklabels(),fontsize=8)
em_fig.tight_layout()
endmember_plot = FigureCanvasTkAgg(em_fig, master=tab1)
endmember_plot.get_tk_widget().grid(column=2,row=1)

# Create plot for mosaic pattern
mosaic_label = tk.Label(tab1, text="Mosaic Pattern", anchor='n')
mosaic_label.grid(column = 1, row = 4, padx = 20, pady = 0)
mosaic_image = tk.Label(tab1,image=img, width=200, height=200)
mosaic_image.grid(column = 1, row = 3, padx = 20, pady = 0) 

# Create plot for mosaic pattern
unmixed_label = tk.Label(tab1, text="Demosaicked and Unmixed Image", anchor='n')
unmixed_label.grid(column = 2, row = 4, padx = 20, pady = 0)
unmixed_image = tk.Label(tab1,image=img, width=200, height=200)
unmixed_image.grid(column = 2, row = 3, padx = 20, pady = 0) 

# Create table to display data
tree = ttk.Treeview(tab1)
tree.grid(column = 3, row = 1, padx = 20, pady = 0)

# Defining number of columns 
tree["columns"] = ("1", "2", "3","4","5") 

# Defining heading 
tree['show'] = 'headings'

# Assigning the width and anchor to  the 
# respective columns 
tree.column("1", width = 75) 
tree.column("2", width = 75) 
tree.column("3", width = 75) 
tree.column("4", width = 75) 
tree.column("5", width = 75) 

def display_column(e):
    global filters, spectra, wavelength, bands, bws
    n = int(e)-1
    if bands is None or bws is None:
        return
    filters = c_[bands[n,:], bws[n,:], ones_like(bands[n,:])]
    filter_responses = generate_filter_data(filters, wavelength)
    plot_endmembers(spectra, wavelength)
    f_select.set(e) # default value
    return

# Assigning the heading names to the  
# respective columns 
columns = ("1","2","3","4","5")
for col in columns:
    tree.heading(col, text ="Set " + col, command=lambda _col=col: display_column(_col)) 

def populate_table(bands,bws):
    tree.delete(*tree.get_children())

    for i in range(bands.shape[1]):
        tree.insert("", 'end', text="A",
            values =[str(bands[j,i]) + "(" + str(bws[j,i]) + ")" for j in range(bands.shape[0])]) 

    return

def plot_unmixed(image, filters, abundance_map, spectra, wavelength):
    filter_responses = generate_filter_data(filters, wavelength)
    # Generate HSI image using filter data
    MSFA_spectral, _ = create_synthetic_hypercube(image, filter_responses, wavelength)

    # First, calculate expected abundance map
    acc, output = compute_unmixing_accuracy(MSFA_spectral, abundance_map, spectra, wavelength, filters[:,0], filters[:,1], return_predicted=True,filtered_input=True)
    
    # Generate new HSI image
    hypercube, _ = create_synthetic_hypercube(output, spectra, wavelength)
    
    # Redefine colorspace to have correct wavelength sampling
    cs = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65, wavelength=wavelength)    
    
    hsi_RGB, alpha_ch = hypercube_to_RGB(hypercube,wavelength,cs, alpha=True)
    img = Image.fromarray(uint8(hsi_RGB*255.)).resize((200,200))
    mask = Image.fromarray(uint8(alpha_ch*255.)).resize((200,200))
    black = Image.new('RGB',(200,200))
    img = Image.composite(img,black,mask)
    img = ImageTk.PhotoImage(img) # convert to PhotoImage
    unmixed_image.config(image=img)
    unmixed_image.image=img
    return

def plot_mosaic(pd, f, pattern):
    mos = zeros((200,200,3))

    for p in arange(pd[0]*pd[1]):
        px = 200 / pd[0] 
        py = 200 / pd[1] 
        i = int(floor(p / pd[0]))
        j = int(p - pd[0]*i)

        try:
            mos[int(i*py):int((i+1)*py)-1,int(j*px):int((j+1)*px)-1,] = cs.spec_to_rgb(f[p])
        except:
            mos[int(i*py):int((i+1)*py)-1,int(j*px):int((j+1)*px)-1,] = (f[p],f[p],f[p])
    img = Image.fromarray(uint8(mos*255.))
    img = ImageTk.PhotoImage(img) # convert to PhotoImage
    mosaic_image.config(image=img)
    mosaic_image.image=img

def plot_endmembers(spectra, wavelength):
    global filters
    a.cla()
    a.set_ylabel("Normalized Signal", fontsize=8)
    a.set_xlabel("Wavelength (nm)", fontsize=8)
    a.set_ylim(0,1.4)
    a.set_xticklabels(a.get_xticklabels(),fontsize=8)
    a.set_yticklabels(a.get_yticklabels(),fontsize=8)

    # Create Legend
    legend_elements = [Line2D([0], [0], color='k',label='Filter'),
                           Line2D([0], [0], linestyle='--', color='k', label='Endmember')]
        
    for i in range(spectra.shape[0]):
        a.plot(wavelength, spectra[i,:]/spectra[i,:].max(), "k--")
        
    if not filters is None:
        filter_responses = generate_filter_data(filters, wavelength)
        for j in range(filter_responses.shape[0]):
            color = cs.spec_to_rgb(filter_responses[j,:])
            a.plot(wavelength,filter_responses[j,:], c=color)
            
    a.set_xticks([wavelength[0],
                  wavelength[int(len(wavelength)*.33)],
                  wavelength[int(len(wavelength)*.67)],
                  wavelength[-1]])
    a.set_yticks([0,0.5,1])
    a.set_xticklabels(["%.1f" % x for x in [wavelength[0],
                                            wavelength[int(len(wavelength)*.33)],
                                            wavelength[int(len(wavelength)*.67)],
                                            wavelength[-1]]],fontsize=8)
    a.set_yticklabels([0,0.5,1],fontsize=8)
    em_fig.tight_layout()
    a.legend(handles=legend_elements,frameon=False, loc='upper center', ncol=2, fontsize=8)
    endmember_plot.draw()
    return
    
def set_data_input():
    '''Callback to set data input for abundance map, noise array, and endmembers.'''
    global fname, a_map, endmembers, noise, wavelength, spectra, cs, wv_max, wv_min, filters, corr, mosaic, pattern_dims, h_ref
    global N_bands,random_sampling,bw_max,bw_min,wv_max,wv_min,bstep,pstep,n_iter,loops,merit_function,demosaicking,op_type,merit_function_spatial,op_type_spectral
    # Get file name and load data.
    # First filter out if the user selected a sub-file.
    fname = filedialog.askopenfilename()
    fname = fname.replace("_corr.npy",".npy")
    fname = fname.replace("_endmembers.npy",".npy")
    fname = fname.replace("_noise.npy",".npy")
    fname = fname.replace("_mosaic.npy",".npy")
    fname = fname.replace("_filters.npy",".npy")
    fname = fname.replace("_params.npy",".npy")
    
    # Reset display
    demosaicked_image.config(image=placeholder)
    demosaicked_image.image=placeholder
    
    disp_image.config(image=placeholder)
    disp_image.image=placeholder
    
    unmixed_image.config(image=placeholder)
    unmixed_image.image=placeholder
    
    plot_mosaic((2,2), [1,1,1,1], [0,1,2,3])
    
    # Try to load data
    if ".mat" in fname:
        hypercube, a_map, endmembers, noise, filters, corr, mosaic = load_reference_data(fname)
        wavelength = endmembers[0,:]
        spectra = endmembers[1:,:]   
        h_ref = hypercube.copy() # Store this for later reference.
    else:
        try:
            a_map, endmembers, noise, filters, corr, mosaic = load_target_data(fname)
            # Construct hypercube
            wavelength = endmembers[0,:]
            spectra = endmembers[1:,:]   
            hypercube, _ = create_synthetic_hypercube(a_map, spectra, wavelength)

        except:
            print("Please provide name to valid data.")
            return
    # Add noise and re-scale
    hypercube += noise
    '''
    # This is a BUG!!!
    # Unless scale down the endmember signature by the same amount to keep abundance map at the same scale!!!
    if hypercube.max() > 1.:
        hypercube /= hypercube.max()
    '''
      
    # Find the minimum and maximum wavlength from the input array
    wv_max = wavelength[-1]
    wv_min = wavelength[0]
        
    # Try to load variables file
    try:
        with open(fname.replace(".npy","_params.npy"), "rb") as f:
            N_bands,random_sampling,bw_max,bw_min,wv_max,wv_min,bstep,pstep,n_iter,loops,op_type_spectral,merit_function,demosaicking,op_type,op_type_spectral,merit_function_spatial = pickle.load(f)
            B_num_bands.delete(0, tk.END)
            B_max_bw.delete(0, tk.END)
            B_min_bw.delete(0, tk.END)
            B_bw_step.delete(0, tk.END)
            B_wv_step.delete(0, tk.END)
            B_iter.delete(0, tk.END)
            B_loops.delete(0, tk.END)
            
            B_num_bands.insert(0, str(N_bands))
            B_max_bw.insert(0, str(bw_max))
            B_min_bw.insert(0, str(bw_min))
            B_bw_step.insert(0, str(bstep))
            B_wv_step.insert(0, str(pstep))
            B_iter.insert(0, str(n_iter))
            B_loops.insert(0, str(loops))
            B_rand_samp.var.set(bool(random_sampling))
            if bool(random_sampling):
                B_rand_samp.var.select()
            else:
                B_rand_samp.var.deselect()
            
            if op_type_spectral == "gradient-descent":
                B_op_function.set("Gradient Descent")
            # Put other options here when implemented
            if merit_function == "unmixing":
                B_merit_function.set("RMS Unmixing Accuracy")
            # Put other options here when implemented
            if op_type == "exhaustive":
                search.set("Exhaustive")
            # Put other options here when implemented
            if demosaicking == "ISD":
                demosaic_algorithm.set("Iterative Spectral Differences")
            elif demosaicking == "SD":
                demosaic_algorithm.set("Spectral Differences")
            else:
                demosaic_algorithm.set("Weighted Bi-Linear Interpolation")
            if merit_function_spatial == "RMS-RGB":
                B_merit_function_spatial.set("RMS RGB Difference")
    except:
        pass

    if not filters is None:
        bands = array([filters[:,0]])
        bws = array([filters[:,1]])
        populate_table(bands,bws)
    
    B_max_wave.delete(0, tk.END)
    B_min_wave.delete(0, tk.END)    
    B_max_wave.insert(0, str(wv_max))
    B_min_wave.insert(0, str(wv_min))
    
    # Redefine colorspace to have correct wavelength sampling
    cs = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65, wavelength=wavelength)    
    
    hsi_RGB, alpha_ch = hypercube_to_RGB(hypercube,wavelength,cs, alpha=True)
    img = Image.fromarray(uint8(hsi_RGB*255.)).resize((200,200))
    mask = Image.fromarray(uint8(alpha_ch*255.)).resize((200,200))
    black = Image.new('RGB',(200,200))
    img = Image.composite(img,black,mask)
    img = ImageTk.PhotoImage(img) # convert to PhotoImage
    disp_image.config(image=img)
    disp_image.image=img
    fname_label.config(text="Data currently selected: " +  os.path.split(fname)[-1])
    plot_endmembers(spectra, wavelength)
    
    if not mosaic is None:
        filter_responses = generate_filter_data(filters, wavelength)
        ordered_filters = array([filter_responses[i] for i in mosaic.reshape(-1)])
        pattern_dims = mosaic.shape
        mosaic_size.set("x".join([str(pattern_dims[0]),str(pattern_dims[1])]))
        plot_mosaic(pattern_dims, ordered_filters, mosaic.reshape(-1))
        
        if corr is None and demosaicking == "ISD":
            corr = generate_correlation_matrix(hypercube,filters,wavelength)
            
        demosaicked = demosaick_input_image(hypercube,wavelength,pattern_dims, ordered_filters, corr, demosaicking=demosaicking)
        msfa_RGB, alpha_ch = MSFA_to_RGB(demosaicked,ordered_filters, wavelength,cs, alpha=True)
        img = Image.fromarray(uint8(msfa_RGB*255.)).resize((200,200))
        mask = Image.fromarray(uint8(alpha_ch*255.)).resize((200,200))
        black = Image.new('RGB',(200,200))
        img = Image.composite(img,black,mask)
        img = ImageTk.PhotoImage(img) # convert to PhotoImage
        demosaicked_image.config(image=img)
        demosaicked_image.image=img
        
        plot_unmixed(demosaicked, filters, a_map, spectra, wavelength)
    return
    
def callback_spectral_bands():
    '''Callback to optimize spectral bands. Returns top 5 filter options.'''
    global bands, bws, filters
    if a_map is None or spectra is None or wavelength is None:
        return
    '''
    Create synthetic hypercube always?
    What if an original noisy hypercube is already input? This should not be overwritten by a synthectic hypercube.
    '''
    hypercube, _ = create_synthetic_hypercube(a_map, spectra, wavelength)
    hypercube += noise
    '''
    # This is a BUG!!!
    # Unless scale down the endmember signature by the same amount to keep abundance map at the same scale!!!
    if hypercube.max() > 1.:
        hypercube /= hypercube.max()
    '''
    bands, bws = optimize_spectral_bands(hypercube,
                                         a_map,spectra,wavelength,
                                         N_bands,
                                         op_type_spectral,
                                         verbose=True,
                                         bw_max = bw_max,
                                         bw_min = bw_min,
                                         wv_max = wv_max,
                                         wv_min = wv_min,
                                         bstep = bstep,
                                         pstep = pstep,
                                         n_iter = n_iter,
                                         loops = loops,
                                         random_sampling=random_sampling)
    
    bands = array(bands)
    bws = array(bws)
    filters = c_[bands[0,:], bws[0,:], ones_like(bands[0,:])]
    filter_responses = generate_filter_data(filters, wavelength)
    populate_table(bands,bws)
    # Delete bands and bws variables since we only have one with uniform sampling
    if not random_sampling:
        bands = None
        bws = None
    f_select.set("1")
    plot_endmembers(spectra, wavelength)
    return

def save_filter_set():
    '''Callback to save a particular filter set'''
    global filters, bands, bws
    if bands is None and filters is None:
        print("Need to generate spectral filters by spectral band optimization.")
        return
    if filters is None:
        n = int(f_select.get())- 1
        filters = c_[bands[n,:], bws[n,:], ones_like(bands[n,:])]
    
    np.save(fname.replace(".npy","_filters.npy").replace(".mat","_filters.npy"),filters)
    return

def select_filter_set(e):
    '''Wrapper for function to plot different set of top filter options.
       Only has an effect when a series of bandwidths and bands are available.'''
    global filters, bands, bws
    if bands is None:
        print("Need to generate spectral filters by spectral band optimization.")
        return
    n = int(f_select.get())- 1
    filters = c_[bands[n,:], bws[n,:], ones_like(bands[n,:])]
    filter_responses = generate_filter_data(filters, wavelength)
    plot_endmembers(spectra, wavelength)
    return

def save_all_as():
    '''Callback to save all data with new name.    '''
    global fname, a_map, endmembers, noise, filters, corr, mosaic
    global  N_bands,random_sampling,bw_max,bw_min,wv_max,wv_min,bstep,pstep,n_iter,loops,merit_function,demosaicking,op_type,merit_function_spatial
    try:
        fname = filedialog.asksaveasfilename()
        file,ext = os.path.splitext(fname)
    except:
        return
    
    params = [N_bands,random_sampling,bw_max,bw_min,wv_max,
              wv_min,bstep,pstep,n_iter,loops,op_type_spectral,merit_function,
              demosaicking,op_type,merit_function_spatial]
    

    with open(file+'_params.npy', "wb") as f:
        pickle.dump(params, f)
        

    if not a_map is None:
        np.save(file+".npy", a_map)

    if not endmembers is None:
        np.save(file+"_endmembers.npy",endmembers)

    if not noise is None:
        np.save(file+"_noise.npy",noise)

    if not filters is None:   
        np.save(file+"_filters.npy",filters)
     
    if not corr is None:   
        np.save(file+"_corr.npy",corr)
      
    if not mosaic is None:      
        np.save(file+"_mosaic.npy",mosaic)
     
    return

def export_to_mat():
    '''Callback to save all data with .mat file scheme name.'''
    global fname, a_map, endmembers, noise, filters, corr, mosaic
    global  N_bands,random_sampling,bw_max,bw_min,wv_max,wv_min,bstep,pstep,n_iter,loops,merit_function,demosaicking,op_type,merit_function_spatial
    
    one_file=True
    
    try:
        fname = filedialog.asksaveasfilename()
        file,ext = os.path.splitext(fname)
    except:
        return
    
    params = [N_bands,random_sampling,bw_max,bw_min,wv_max,
              wv_min,bstep,pstep,n_iter,loops,op_type_spectral,merit_function,
              demosaicking,op_type,merit_function_spatial]
    
    if one_file:
        savemat(file+".mat", {
            "abundance_map": a_map,
            "endmembers": endmembers,
            "noise": noise,
            "filters": filters,
            "correlation_matrix": corr,
            "mosaic": mosaic,
            "parameters": params
                        })
        return

    with open(file+'_params.txt', "wb") as f:
        pickle.dump(params, f)

    if not a_map is None:
        savemat(file+".mat", {"abundance_map": a_map})

    if not endmembers is None:
        savemat(file+"_endmembers.mat", {"endmembers": endmembers})

    if not noise is None:
        savemat(file+"_noise.mat",{"noise": noise})

    if not filters is None:   
        savemat(file+"_filters.mat",{"filters": filters})
     
    if not corr is None:   
        savemat(file+"_corr.mat",{"correlation_matrix": corr})
      
    if not mosaic is None:      
        savemat(file+"_mosaic.mat",{"mosaic": mosaic})
     
    return

def save_mosaic():
    global mosaic
    if mosaic is None:
        print("Need to generate mosaic before saving")
    np.save(fname.replace(".npy","_mosaic.npy").replace(".mat","_mosaic.npy"),mosaic)
    return
    
def callback_mosaic():
    global a_map, spectra, wavelength, mosaic, pattern_dims, filters, corr, h_ref
    
    if filters is None and bands is None:
        print("Need to generate spectral filters before mosaic optimization.")
        return
    
    if filters is None: # Default to top option
        filters = c_[bands[0,:], bws[0,:], ones_like(bands[0,:])]
        
    filter_responses = generate_filter_data(filters, wavelength)
    print(pattern_dims, demosaicking, op_type, merit_function_spatial)    
    
    if h_ref is not None:
        hypercube = h_ref.copy()
    else:
        hypercube, _ = create_synthetic_hypercube(a_map, spectra, wavelength)
        if corr is None:
            corr = generate_correlation_matrix(hypercube,filter_responses,wavelength)

        hypercube += noise
        
    '''
    # This is a BUG!!!
    # Unless scale down the endmember signature by the same amount to keep abundance map at the same scale!!!
    if hypercube.max() > 1.:
        hypercube /= hypercube.max()
    '''

    mosaic_unravel, demosaicked = spatial_optimization_MSFA(hypercube,
                                                    wavelength,
                                                    pattern_dims,
                                                    filter_responses,
                                                    corr,
                                                    filter_labels=None,
                                                    demosaicking=demosaicking,
                                                    op_type = op_type,
                                                    merit_function=merit_function_spatial,
                                                    verbose=True)
    ordered_filters = array([filter_responses[i] for i in mosaic_unravel])

    msfa_RGB, alpha_ch = MSFA_to_RGB(demosaicked,ordered_filters, wavelength,cs, alpha=True)
    img = Image.fromarray(uint8(msfa_RGB*255.)).resize((200,200))
    mask = Image.fromarray(uint8(alpha_ch*255.)).resize((200,200))
    black = Image.new('RGB',(200,200))
    img = Image.composite(img,black,mask)
    
    img = ImageTk.PhotoImage(img) # convert to PhotoImage
    demosaicked_image.config(image=img)
    demosaicked_image.image=img
    plot_mosaic(pattern_dims, ordered_filters, mosaic_unravel)
    mosaic = array(mosaic_unravel).reshape(pattern_dims)
    
    plot_unmixed(demosaicked, filters, a_map, spectra, wavelength)
    return

load_data = tk.Button(tab1, width=10, height=1, bg="black", fg="white", text="Load Data",command=set_data_input)
load_data.grid(column = 0, row = 0, padx = 0, pady = 20)  

f_select_label = tk.Label(tab1, text="Select Filter Set: ", anchor='w')
f_select_label.grid(column = 2, row = 0, padx = 20, pady = 0) 
filter_select_options = ["1", "2", "3","4","5"]
f_select = tk.StringVar(tab3)
f_select.set(filter_select_options[0]) # default value
select_filters = tk.OptionMenu(tab1, f_select, *filter_select_options, command=select_filter_set)
select_filters.configure(width=10)
select_filters.grid(column=3,row=0,padx=0,pady=20)

save_filters = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Save Selected Filter Set",command=save_filter_set)
save_filters.grid(column = 3, row = 2, padx = 10, pady = 20)  

optimize_spectral = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Optimize Spectral Bands", command=callback_spectral_bands)
optimize_spectral.grid(column = 1, row = 1, padx = 30, pady = 30)   

save_mosaic = tk.Button(tab1, width=10, height=1, bg="black", fg="white", text="Save Mosaic",command=save_mosaic)
save_mosaic.grid(column = 1, row = 4, padx = 10, pady = 20)  

optimize_spatial = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Optimize Mosaic", command=callback_mosaic)
optimize_spatial.grid(column = 1, row = 2, padx = 30, pady = 20)   

save_as = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Save Data As",command=save_all_as)
save_as.grid(column = 3, row = 4, padx = 10, pady = 20)  


# Tab 2: Spectral band optimization options
ttk.Label(tab2, text ="Select spectral band and optimization options").grid(column = 0, row = 0,  padx = 30, pady = 30) 
ttk.Label(tab2, text ="Number of Bands").grid(column = 0, row = 1, padx = 30, pady = 20) 
ttk.Label(tab2, text ="Minimum Wavelength (nm)").grid(column = 0, row = 2, padx = 10, pady = 5) 
ttk.Label(tab2, text ="Maximum Wavelength (nm)").grid(column = 0, row = 3, padx = 10, pady = 5) 
ttk.Label(tab2, text ="Wavelength Step (nm)").grid(column = 0, row = 4, padx = 10, pady = 5) 
ttk.Label(tab2, text ="Minimum Bandwidth (nm)").grid(column = 0, row = 5, padx = 10, pady = 5) 
ttk.Label(tab2, text ="Maximum Bandwidth (nm)").grid(column = 0, row = 6, padx = 10, pady = 5) 
ttk.Label(tab2, text ="Bandwidth Step (nm)").grid(column = 0, row = 7, padx = 10, pady = 5) 
ttk.Label(tab2, text ="# Stochastic Iterations").grid(column = 0, row = 8, padx = 10, pady = 5) 
ttk.Label(tab2, text ="# Gradient Descent Loops").grid(column = 0, row = 9, padx = 10, pady = 5) 
ttk.Label(tab2, text ="Random Sampling (Uncheck for uniform)").grid(column = 0, row = 10, padx = 30, pady = 5) 
ttk.Label(tab2, text ="Optimization Type").grid(column = 0, row = 11, padx = 30, pady = 5) 
ttk.Label(tab2, text ="Merit Function").grid(column = 0, row = 12, padx = 30, pady = 5) 

# Number of bands
B_num_bands = tk.Entry(tab2)
B_num_bands.grid(column = 1, row = 1, padx = 10, pady = 5)
# Minimum wavelength
B_min_wave = tk.Entry(tab2)
B_min_wave.grid(column = 1, row = 2, padx = 10, pady = 5) 
# Maximum wavelength
B_max_wave = tk.Entry(tab2)
B_max_wave.grid(column = 1, row = 3, padx = 10, pady = 5) 
# Wavelength step
B_wv_step = tk.Entry(tab2)
B_wv_step.grid(column = 1, row = 4, padx = 10, pady = 5) 
# Minimum bandwidth
B_min_bw = tk.Entry(tab2)
B_min_bw.grid(column = 1, row = 5, padx = 10, pady = 5) 
# Maximum bandwidth
B_max_bw = tk.Entry(tab2)
B_max_bw.grid(column = 1, row = 6, padx = 10, pady = 5) 
# Bandwidth_step
B_bw_step = tk.Entry(tab2)
B_bw_step.grid(column = 1, row = 7, padx = 10, pady = 5) 
# Stochastic iterations
B_iter = tk.Entry(tab2)
B_iter.grid(column = 1, row = 8, padx = 10, pady = 5) 
# Gradient Descent Loops
B_loops = tk.Entry(tab2)
B_loops.grid(column = 1, row = 9, padx = 10, pady = 5) 
# Random Sampling
rs_var = tk.IntVar()
B_rand_samp = tk.Checkbutton(tab2, variable=rs_var)
B_rand_samp.grid(column = 1, row = 10, padx = 10, pady = 5)
B_rand_samp.var = rs_var

# Merit Function
op_options = ["Gradient Descent", "Simulated Annealing", "Genetic Algorithm (not active)"]
B_op_function = tk.StringVar(tab2)
B_op_function.set(op_options[0]) # default value
op_menu = tk.OptionMenu(tab2, B_op_function, *op_options)
op_menu.configure(width=25)
op_menu.grid(column=1,row=11,padx=10,pady=5)

# Merit Function
mf_options = ["RMS Unmixing Accuracy", "Option 2", "Option 3"]
B_merit_function = tk.StringVar(tab2)
B_merit_function.set(mf_options[0]) # default value
mf_menu = tk.OptionMenu(tab2, B_merit_function, *mf_options)
mf_menu.configure(width=25)
mf_menu.grid(column=1,row=12,padx=10,pady=5)

def set_spectral_options():
    '''Callback to set spectral band optimization options''' 
    global N_bands, bw_max, bw_min, wv_max, wv_min, bstep,pstep,n_iter,loops,merit_function, random_sampling,op_type_spectral
    N_bands = int(B_num_bands.get())
    bw_max = float(B_max_bw.get())
    bw_min = float(B_min_bw.get())
    wv_max = float(B_max_wave.get())
    wv_min = float(B_min_wave.get())
    bstep = float(B_bw_step.get())
    pstep = float(B_wv_step.get())
    n_iter = int(B_iter.get())
    loops = int(B_loops.get())
    random_sampling = bool(B_rand_samp.var.get())
    mf = B_merit_function.get()
    if mf == "RMS Unmixing Accuracy":
        merit_function = "unmixing"
    else:
        merit_function = "unmixing"
    op = B_op_function.get()
    if op == "Gradient Descent":
        op_type_spectral = "gradient-descent"
    elif op == "Simulated Annealing":
        op_type_spectral = "simulated-annealing"
    else:
        op_type_spectral = "gradient-descent"
        
        
    print(N_bands, bw_max, bw_min, wv_max, wv_min, bstep,pstep,n_iter,loops,op_type_spectral,merit_function, random_sampling)

set_spectral_ops = tk.Button(tab2, width=10, height=1, bg="black", fg="white", text="Set Options", command=set_spectral_options)
set_spectral_ops.grid(column = 0, row = 13, padx = 30, pady = 10)   

# Tab 3: Spatial Optimization options
ttk.Label(tab3, text ="Select spatial mosaic and optimization options").grid(column = 0,row = 0,padx = 30,pady = 30) 

ttk.Label(tab3, text ="Mosaic Size").grid(column = 0, row = 1,  padx = 30, pady = 30) 
ttk.Label(tab3, text ="Demosaicking Algorithm").grid(column = 0, row = 2,  padx = 30, pady = 30) 
ttk.Label(tab3, text ="Search Algorithm").grid(column = 0, row = 3,  padx = 30, pady = 30) 
ttk.Label(tab3, text ="Merit Function").grid(column = 0, row = 4,  padx = 30, pady = 30) 

mosaic_size_options = ["2x2","3x3","4x4","2x3", "3x2","4x2","2x4","3x4","4x3"]
mosaic_size = tk.StringVar(tab3)
mosaic_size.set(mosaic_size_options[0]) # default value
mosaic_size_menu= tk.OptionMenu(tab3,mosaic_size, *mosaic_size_options)
mosaic_size_menu.configure(width=35)
mosaic_size_menu.grid(column=1,row=1,padx=10,pady=20)

dm_options = ["Weighted Bi-Linear Interpolation", "Iterative Spectral Differences", "Spectral Differences"]
demosaic_algorithm = tk.StringVar(tab3)
demosaic_algorithm.set(dm_options[0]) # default value
dm_menu= tk.OptionMenu(tab3, demosaic_algorithm, *dm_options)
dm_menu.configure(width=35)
dm_menu.grid(column=1,row=2,padx=10,pady=20)

search_options = ["Exhaustive", "Simulated Annealing (Not active)", "Genetic Algorithm (Not active)"]
search = tk.StringVar(tab3)
search.set(search_options[0]) # default value
search_menu = tk.OptionMenu(tab3, search, *search_options)
search_menu.configure(width=35)
search_menu.grid(column=1,row=3,padx=10,pady=20)

# Merit Function
mf_spatial_options = ["RMS Pixel Difference", "RMS RGB Difference", "DFT (Not active)"]
B_merit_function_spatial = tk.StringVar(tab3)
B_merit_function_spatial.set(mf_spatial_options[0]) # default value
mf_spatial_menu = tk.OptionMenu(tab3, B_merit_function_spatial, *mf_spatial_options)
mf_spatial_menu.configure(width=35)
mf_spatial_menu.grid(column=1,row=4,padx=10,pady=5)

def set_spatial_options():
    '''Callback to set spatial band optimization options''' 
    global pattern_dims, demosaicking, op_type, merit_function_spatial
    
    dm = demosaic_algorithm.get()
    if dm == "Iterative Spectral Differences":
        demosaicking = "ISD"
    elif dm == "Spectral Differences":
        demosaicking = "SD"
    else:
        demosaicking = "WB"
    
    pd = mosaic_size.get()
    pattern_dims = tuple([int(p) for p in pd.split("x")])
    
    ot = search.get()
    if ot == "Exhaustive":
        op_type = "exhaustive"
    else:
        op_type = "exhaustive"
        
    mfs = B_merit_function_spatial.get()
    if mfs == "RMS RGB Difference":
        merit_function_spatial = 'RMS-RGB'
    else:
        merit_function_spatial = 'RMS'
        
    print(pattern_dims, demosaicking, op_type, merit_function_spatial)
    
set_spatial_ops = tk.Button(tab3, width=10, height=1, bg="black", fg="white", text="Set Options", command=set_spatial_options)
set_spatial_ops.grid(column = 0, row = 5, padx = 30, pady = 10)   

# Final commands to pack everything
tabControl.pack(expand = 1, fill ="both") # This causes window to become as small as possible.

# Need to pre-populate buttons with default values after packing
B_num_bands.insert(0, str(N_bands))
B_max_bw.insert(0, str(bw_max))
B_min_bw.insert(0, str(bw_min))
B_max_wave.insert(0, str(300))
B_min_wave.insert(0, str(700))
B_bw_step.insert(0, str(bstep))
B_wv_step.insert(0, str(pstep))
B_iter.insert(0, str(n_iter))
B_loops.insert(0, str(loops))
B_rand_samp.select()

plot_mosaic((2,2), [1,1,1,1], [0,1,2,3])


# Run Loops
root.mainloop() 
root = tk.Tk()
app = Application(master=root)

app.mainloop()
