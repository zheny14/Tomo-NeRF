# Tomo-NeRF

Tomo-NeRF is a tool that allows you to perform ultra-sparse view tomographic reconstruction. This repository provides the necessary files for training, optimization, and testing of the Tomo-NeRF model.

### Usage: 
#### Data Preparation: 
Tomo-NeRF requires a set of 2D radiographic projection images. Ensure that you have a set of 2-6 images, evenly spaced from 0 degrees to 180 degrees.  Here we offers a set of real x-ray imaging and artificial image samples in compressed files (unzip before use).

#### Training
Specifiy DATAPATH with the path to the directory containing your projection images. Adjust the parameter as requested in script based on the number of images/types of images you have.
The training script will save the trained model for each epoch. Please note here we only provide part of training samples due to the limitation of file size. You can access the fully trained model through the following link: [Trained Model](https://drive.google.com/drive/folders/1-uTtm3OzJTFJs3P851HDodRUjEA_llNw?usp=sharing)

#### Optimization
After training, you can optimize the Tomo-NeRF model on the testing samples (without the knowlege of 3D information) using the optimization script. Specifiy DATAPATH with the path to the directory containing your projection images and load the trained model (Example:[Trained Model](https://drive.google.com/drive/folders/1-uTtm3OzJTFJs3P851HDodRUjEA_llNw?usp=sharing)). Adjust the parameter as requested in script based on the number of images/types of images you have.
The optimization script will save the optimized model for each epoch.

#### Testing and Visualization
Once the optimization is complete, you can use the model to perform tomographic reconstruction on new projection images with the script in the Testing folder. Specifiy DATAPATH with the path to the directory containing your projection images and load the trained and optimized model (Example:[Trained Model](https://drive.google.com/drive/folders/1-uTtm3OzJTFJs3P851HDodRUjEA_llNw?usp=sharing)). The script will save the reconstructed 3D images as tif files. 

### Development
If you are interested in trying new architectures, see the class of Net_upsampling in the scripts.

### Versions

Version 1.0: [Ultra-sparse View X-ray Computed Tomography for 4-D Imaging](https://doi.org/10.26434/chemrxiv-2023-3qrhl)
