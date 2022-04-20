# Import relevant GUI packages
import tkinter as tk
import os, pickle
from tkinter import ttk, Canvas, PhotoImage, filedialog, Scale, Toplevel, DoubleVar
from PIL import Image, ImageTk
from matplotlib.lines import Line2D
import numpy
import pandas

# Import plotting packages
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import MSFA Optimization Code
# import msfa_optimization
# from msfa_optimization import *

root = tk.Tk() 
root.title("MSFA Optimizer Tool: Input Hypercube Generator") 
root.geometry("1200x700")

tabControl = ttk.Notebook(root) 
  
tab1 = ttk.Frame(tabControl) 
tab2 = ttk.Frame(tabControl) 
tab3 = ttk.Frame(tabControl) 

# Define global variables
fname = "../input/USAF.png"
fname_em = ""
base = None # Full resolution base image
a_map = None # Output abundance map
endmembers = None
spectra = None
wavelength = None
noise = None
newWindow = None
rect_bounds = None
em_sliders = []
noise_SNR = 0.

img = Image.open(fname)  # PIL solution
img = img.resize((200, 200), Image.ANTIALIAS) #The (250, 250) is (height, width)
img = ImageTk.PhotoImage(img) # convert to PhotoImage

topx, topy, botx, boty = 0, 0, 0, 0
rect_id = None

def get_mouse_posn(event):
    global topy, topx

    topx, topy = event.x, event.y

def update_sel_rect(event):
    global rect_id
    global topy, topx, botx, boty

    botx, boty = event.x, event.y
    disp_image.coords(rect_id, topx, topy, botx, boty)  # Update selection rect.

disp_image = tk.Canvas(tab1, width=img.width(), height=img.height(),
                   borderwidth=0, highlightthickness=0)

disp_image.grid(column = 0, row = 1, padx = 20, pady = 0) 
disp_image.img = img  # Keep reference in case this code is put into a function.
disp_image.create_image(0, 0, image=img, anchor=tk.NW)

# Create selection rectangle (invisible since corner points are equal).
rect_id = disp_image.create_rectangle(topx, topy, topx, topy,
                                  dash=(2,2), fill='', outline='white')

disp_image.bind('<Button-1>', get_mouse_posn)
disp_image.bind('<B1-Motion>', update_sel_rect)

noise_img = Image.open(fname)  # PIL solution
noise_img = noise_img.resize((200, 200), Image.ANTIALIAS) #The (250, 250) is (height, width)
noise_img = ImageTk.PhotoImage(noise_img) # convert to PhotoImage
#tabControl.insert(tab1,tab1,image=img)
  
tabControl.add(tab1, text ='Hypercube Creation Tool') 
tabControl.add(tab2, text ='Spatial Diagnostics') 
tabControl.add(tab3, text ='Spectral Diagnostics') 
    
# Tab 1: Main window for optimization
ttk.Label(tab1, text ="Multispectral Filter Array Optimizer")

# Tab 1: Main window for optimization
# Call grid on separate line to avoid NoneType definition
fname_label = tk.Label(tab1, text = "Data currently selected: " +fname, width=50, anchor="w")
fname_label.grid(column = 1, row = 0, padx = 0, pady = 20) 

# Tab 1: Main window for optimization
# Call grid on separate line to avoid NoneType definition
fname_em_label = tk.Label(tab1, text = "Endmembers currently selected: " +fname, width=50, anchor="w")
fname_em_label.grid(column = 3, row = 0, padx = 0, pady = 20) 

# Note that I will eventually have to change these to canvases since we'll be plotting
# data, not just displaying images.
#disp_image_label = tk.Label(tab1, text="Greyscale Input image", anchor='n')
#disp_image_label.grid(column = 0, row = 2, padx = 20, pady = 0) 
#disp_image = tk.Label(tab1,image=img, width=200, height=200)
#disp_image.grid(column = 0, row = 1, padx = 20, pady = 0) 

hsi_img_label = tk.Label(tab1, text="RGB-Equivalent of HSI Image", anchor='n')
hsi_img_label.grid(column = 0, row = 4, padx = 20, pady = 0) 
hsi_image = tk.Label(tab1 ,image=img, width=200, height=200,anchor='s')
hsi_image.grid(column = 0, row = 3, padx = 20, pady = (20,0)) 

# Create plot for endmembers
endmember_label = tk.Label(tab1, text="Endmembers", anchor='n')
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
noise_label = tk.Label(tab1, text="Noised Image", anchor='n')
noise_label.grid(column = 2, row = 4, padx = 20, pady = 0)
noise_image = tk.Label(tab1, image=noise_img, width=200, height=200)
noise_image.grid(column = 2, row = 3, padx = 20, pady = 0) 

def load_image():
    global fname, base, img, a_map, spectra, wavelength, disp_image, rect_id
    fname = filedialog.askopenfilename()
    try:
        if ".npy" in fname: # If provided as a .npy array
            base = np.load(fname).astype(numpy.float)
            try:
                rgb_weights = array([0.2989, 0.5870, 0.1140])
                base = numpy.dot(base[:,:,:3], rgb_weights)
            except:
                pass
        else:
            base = numpy.array(Image.open(fname)).astype(numpy.float)
            try:
                rgb_weights = array([0.2989, 0.5870, 0.1140])
                base = numpy.dot(base[:,:,:3], rgb_weights)
            except:
                pass
        base /= base.max()
        img = Image.fromarray(uint8(base*255.)).convert('LA')
        img = img.resize((200,200)).convert('LA')
        img = ImageTk.PhotoImage(img) # convert to PhotoImage, greyscale if needed
    except:
        print("Please provide valid greyscale image")
        return

    disp_image.img = img  # Keep reference in case this code is put into a function.
    disp_image.create_image(0, 0, image=img, anchor=tk.NW)
    disp_image.bind('<Button-1>', get_mouse_posn)
    disp_image.bind('<B1-Motion>', update_sel_rect)
    rect_id = disp_image.create_rectangle(0,0,0,0,dash=(2,2), fill='', outline='white')
    
    if spectra is not None:
        N,M = base.shape
        L,Q = spectra.shape
        # Create abundance map. 
        a_map = zeros((N,M,L))
        for i in range(L):
            a_map[:,:,i] = base
        
    generate_hsi_image()
    generate_noise_array()
    fname_label.config(text="Data currently selected: " + os.path.split(fname)[-1])
    return

# function to open a new window  
# on a button click 
def openNewWindow(): 
    global newWindow, spectra, labels, em_sliders
    
    if newWindow is not None:
        newWindow.destroy()
    # Toplevel object which will  
    # be treated as a new window 
    newWindow = Toplevel(root) 
  
    # sets the title of the 
    # Toplevel widget 
    newWindow.title("Endmember Abundance Selector") 
  
    # sets the geometry of toplevel 
    newWindow.geometry("400x600") 
    
    L,Q = spectra.shape
    em_sliders = []
    
    for i in range(L):
        em_sliders.append(DoubleVar())
        s = Scale(newWindow, from_=0, to=1, orient="horizontal", resolution=0.005, length=300, label=labels[i], variable=em_sliders[i])
        s.grid(column = 0, row = i+1, padx = 0, pady = 5) 
        s.set(1)
    
    return

def load_endmembers():
    global endmembers, spectra, wavelength, fname_em, a_map, labels
    fname_em = filedialog.askopenfilename()
    
    try:
        if ".npy" in fname_em:
            endmembers = np.load(fname_em)
        else:
            if ".xls" in fname_em or ".xlsx" in fname_em:
                endmembers = pandas.read_excel(fname_em).values.T
                labels = pandas.read_excel(fname_em).columns.values[1:]
            else:
                endmembers = pandas.read_csv(fname_em).values.T
                labels = pandas.read_csv(fname_em).columns.values[1:]
    except:
        print("Please input valid endmember data in npy, excel, or csv format")
        return
    
    wavelength = endmembers[0,:]
    spectra = endmembers[1:,:]   
    L,Q = spectra.shape
    
    if base is not None:
        N,M = base.shape
        # Create abundance map. 
        a_map = zeros((N,M,L))
 
    a.cla()
    a.set_ylabel("Normalized Signal", fontsize=8)
    a.set_xlabel("Wavelength (nm)", fontsize=8)
    a.set_ylim(0,1.4)
    a.set_xticklabels(a.get_xticklabels(),fontsize=8)
    a.set_yticklabels(a.get_yticklabels(),fontsize=8)
        
    for i in range(L):
        spectra[i,:] /= spectra[i,:].max()
        if base is not None:
            a_map[:,:,i] = base
        if len(labels):
            a.plot(wavelength, spectra[i,:], label=labels[i])
        else:
            a.plot(wavelength, spectra[i,:], label=str(i))
    a.set_xticks([wavelength[0],wavelength[int(len(wavelength)/2)],wavelength[-1]])
    a.set_yticks([0,0.5,1])
    a.set_xticklabels([wavelength[0],wavelength[int(len(wavelength)/2)],wavelength[-1]],fontsize=8)
    a.set_yticklabels([0,0.5,1],fontsize=8)
    em_fig.tight_layout()
    a.legend(frameon=False, loc='upper right', fontsize=8)
    endmember_plot.draw()
       
    fname_em_label.config(text="Endmembers currently selected: " +  os.path.split(fname_em)[-1])
    generate_hsi_image()
    
    openNewWindow()
    return

def save_all_as():
    '''Callback to save all data with new name.    '''
    global a_map, endmembers, noise
    try:
        outname = filedialog.asksaveasfilename()
        file,ext = os.path.splitext(outname)
    except:
        return
        
    if not endmembers is None:
        np.save(file+"_endmembers.npy",endmembers)
    else:
        print("Must select endmembers for data set")
        return

    if not a_map is None:
        np.save(file+".npy", a_map)

    if not noise is None:
        np.save(file+"_noise.npy",noise.reshape((noise.shape[0],noise.shape[1],1)))
    else:
        np.save(file+"_noise.npy",zeros((base.shape[0],base.shape[1],1)))
     
    return

def generate_hsi_image():
    global a_map, spectra, wavelength, img
    
    if a_map is None or spectra is None or wavelength is None:
        # Incomplete data. Need to display standard image
        hsi_image.config(image=img)
        hsi_image.image=img
        return

    hsi,_ = create_synthetic_hypercube(a_map,spectra,wavelength)
    
    # Redefine colorspace to have correct wavelength sampling
    cs = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65, wavelength=wavelength)    
    
    hsi_RGB, alpha_ch = hypercube_to_RGB(hsi,wavelength,cs, alpha=True)
    hsi_img = Image.fromarray(uint8(hsi_RGB*255.))
    hsi_img = hsi_img.resize((200, 200), Image.ANTIALIAS)
    mask = Image.fromarray(uint8(alpha_ch*255.)).resize((200,200))
    black = Image.new('RGB',(200,200))
    hsi_img = Image.composite(hsi_img,black,mask)
    
    hsi_img = ImageTk.PhotoImage(hsi_img)
    hsi_image.config(image=hsi_img)
    hsi_image.image=hsi_img
    return

def generate_noise_array():
    global noise, noise_SNR, noise_img, base
    
    if noise_SNR == 0:
        noise = zeros_like(base)
    else:
        noise = numpy.random.uniform(low=0,high=noise_SNR/5,size=base.shape)
    
    total = base + noise
    total /= total.max()
    noise_img = Image.fromarray(uint8(total*255.)).convert('LA')
    noise_img = noise_img.resize((200, 200), Image.ANTIALIAS)
    noise_img = ImageTk.PhotoImage(noise_img)
    noise_image.config(image=noise_img)
    noise_image.image=noise_img
    return

def set_SNR():
    global noise_SNR
    noise_SNR = float(SNR.get())
    generate_noise_array()
    return

def select_all_pixels():
    global em_sliders, topy, topx, botx, boty
    botx = 200.
    boty = 200.
    disp_image.coords(rect_id, topx, topy, botx, boty)  # Update selection rect.
    return
    
def set_data():
    global a_map, spectra, wavelength
    global em_sliders, topy, topx, botx, boty
    
    y_scale = base.shape[0] / 200.
    x_scale = base.shape[1] / 200.
    
    for i in range(len(em_sliders)):
        values = base[int(topy*y_scale):int(boty*y_scale),int(topx*x_scale):int(botx*x_scale)]*em_sliders[i].get()
        bkgd_values = a_map.copy()
        a_map[int(topy*y_scale):int(boty*y_scale),int(topx*x_scale):int(botx*x_scale),i] = values
        
        # Reset background values.
        for i in range(a_map.shape[2]):
            a_map[:,:,i][base==0] = bkgd_values[:,:,i][base==0]
        
    generate_hsi_image()
    return

def set_background():
    global a_map, spectra, wavelength
    global em_sliders
    
    for i in range(len(em_sliders)):
        a_map[:,:,i][base==0] = em_sliders[i].get()
    
    generate_hsi_image()
    return

def set_uniform():
    global a_map, spectra, wavelength
    global em_sliders
    
    for i in range(len(em_sliders)):
        a_map[:,:,i] += em_sliders[i].get()
        
    generate_hsi_image()
    return
    
set_data = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Set Selected Data",command=set_data)
set_data.grid(column = 1, row = 1, padx = 0, pady = 20)  

set_bkgd = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Set Black Background",command=set_background)
set_bkgd.grid(column = 1, row = 2, padx = 0, pady = 20)  

set_all = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Add Uniform Signal",command=set_uniform)
set_all.grid(column = 1, row = 3, padx = 0, pady = 20)  

load_data = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Select Base Image",command=load_image)
load_data.grid(column = 0, row = 0, padx = 0, pady = 20)  

load_endmembers = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Select Endmembers",command=load_endmembers)
load_endmembers.grid(column = 2, row = 0, padx = 0, pady = 20)  

select_all = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Select All Data",command=select_all_pixels)
select_all.grid(column = 0, row = 2, padx = 0, pady = 20)  

SNR = tk.Entry(tab1)
SNR.grid(column = 3, row = 3, padx = 10, pady = 5) 
SNR.insert(0, noise_SNR)

B_SNR = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Set Noise Level",command=set_SNR)
B_SNR.grid(column = 3, row = 4, padx = 0, pady = 0)  

save_as = tk.Button(tab1, width=20, height=1, bg="black", fg="white", text="Save Data",command=save_all_as)
save_as.grid(column = 1, row = 4, padx = 10, pady = 20)  

tabControl.pack(expand = 1, fill ="both") # This causes window to become as small as possible.
root.mainloop() 
root = tk.Tk()
app = Application(master=root)
app.mainloop()