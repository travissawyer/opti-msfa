# Opti-MSFA: A toolbox for generalized design and optimization of multispectral filter arrays
--------------------------------------------------------------------

**Contributors**: 
Travis W. Sawyer (1)**, Michaela Taylor-Williams (2), Ran Tao (2,3), Ruqiao Xia (2,3), Calum Williams (2), Sarah E. Bohndiek (2,4)

1 - Wyant College of Optical Sciences, University of Arizona, Tucson, USA
2 - Department of Physics, Cavendish Laboratory, University of Cambridge, JJ Thomson Avenue, Cambridge, CB3 0HE, UK
3 - Department of Engineering, Electrical Engineering Division, University of Cambridge, JJ Thomson Avenue, Cambridge, CB3 0FA, UK
4 - Cancer Research UK Cambridge Institute, University of Cambridge, 
Robinson Way, Cambridge, CB2 0RE, UK

** Corresponding author: tsawyer9226@email.arizona.edu

**Python version**: 3.X

**Modules required**: Numpy, MatplotLib, OpenCV (CV2) pysptools (https://pysptools.sourceforge.io/installation.html)

--------------------------------------------------------------------
**Summary**: 

Multispectral imaging captures spatial information across a set of discrete spectral channels and is widely utilized across diverse applications such as remote sensing, industrial inspection and biomedical imaging. Multispectral filter arrays (MSFAs) are integrated filter mosaics which facilitate cost-effective, compact and snapshot multispectral imaging. With MSFAs pre-configured based on application—with selected channels corresponding to targeted absorption spectra—the design of optimal MSFAs is a subject of great interest. Many design and optimization approaches have been introduced for spectral filter selection and spatial arrangement, however, there are few robust approaches for joint spectral-spatial optimization. The techniques are only applicable to limited datasets, and most critically are not available for use, and improvement, from the wider community. Here, we assess current MSFA design techniques and  present Opti-MSFA: A Python-based open-access toolbox for the centralized design and optimization of MSFAs. Opti-MSFA incorporates established spectral-spatial optimization algorithms such as gradient descent and simulated annealing, multispectral-RGB image reconstruction, and is applicable to user-defined input spatial-spectral datasets or imagery. We validate the toolbox against standardized hyperspectral datasets and further show its utility on experimentally acquired fluorescence data. In conjunction with end-user input and collaboration, we foresee the continued development of Opti-MSFA for the benefit of the wider researcher community. Ultimately, we envisage this communal toolbox to offer users the ability to both determine optimal MSFAs for their targeted applications and benchmark new optimization algorithms, such as machine learning-based, against existing approaches.
