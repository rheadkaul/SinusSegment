SinusSegment is an open segmentation algorithm adapted from UNet++ for the purpose of segmentation computed tomography paranasal sinus imaging. 
In our study, we segmented all of the five paranasal sinuses (maxillary, anterior ethmoid, posterior ethmoid, sphenoid and frontal) along with the nasal cavity (anteriorly in line from the nasal bones to the anterior nasal spine and posteriorly in a line from the posterior wall of the sphenoid sinus to the posterior end of the hard palate). 
Our paper explaining the application is linked below: (INSERT link here) 


UNet ++ algorithm used as per Zhou et al 2018 

UNet++: A Nested U-Net Architecture for Medical Image Segmentation
Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang
Arizona State University
Deep Learning in Medical Image Analysis (DLMIA) 2018. (Oral)
Original algortihm available on: https://github.com/MrGiovanni/UNetPlusPlus 


Explanation

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
