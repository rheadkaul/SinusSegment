SinusSegment is an open segmentation algorithm adapted from UNet++ for the purpose of segmenting computed tomography paranasal sinus imaging. 
In our study, we segmented all of the five paranasal sinuses (maxillary, anterior ethmoid, posterior ethmoid, sphenoid and frontal) along with the nasal cavity (anteriorly in a line from the nasal bones to the anterior nasal spine and posteriorly in a line from the posterior wall of the sphenoid sinus to the posterior end of the hard palate). 
Our paper explaining the application is linked below: (INSERT link here) 


UNet ++ algorithm used as per Zhou et al 2018 

UNet++: A Nested U-Net Architecture for Medical Image Segmentation
Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang
Arizona State University
Deep Learning in Medical Image Analysis (DLMIA) 2018. (Oral)
Original algortihm available on: https://github.com/MrGiovanni/UNetPlusPlus 




Explanation

**Pre-processing: **
1. Convert DICOM images to NIFTI files (can use software such as 3D slicer or codes)
2. Choose pre-processing code/
Adapted from MedSAM link available: https://github.com/bowang-lab/MedSAM
Ma, J., He, Y., Li, F. et al. Segment anything in medical images. Nat Commun 15, 654 (2024). https://doi.org/10.1038/s41467-024-44824-z
Article on: https://www.nature.com/articles/s41467-024-44824-z
4. Ensure windowing set to level of 300 and width of 3000

Input: DICOM files of CT paranasal sinuses (bony axial windows with 0.625mm slices) 
Output: NIFTI (.nii) 

**Training/Testing: **

Choose a split of your data for training and testing i.e. 80:20

Input: NIFTI (.nii) 

Output: NIFTI (.nii) 

Folders:

CT_train and Mask_train: place the CT volumes and Segmentation volumes for training into these folders separately.
CT_test and Mask_test: put the CT volumes and Segmentation volumes for testing into these folders separately.

Please note that the Name of your files putted in to XX_train and XX_test should be same.
For example: If you have a CT scan named E1.nii in CT_train, have to also name Segmentation volume E1.nii in Mask_train.

Parameters: contains the parameter of network.
Seg_prediction: will save the result of prediction from network.

Files:

Training.py: training network and a save the parameters into folder Parameters
Testing.py: output the segmentation result and save them into folder Seg_prediction.
Load_dataset.py: loading training and testing dataset.
UnetPlusPlus.py: architecture of UnetPlusPlus.
