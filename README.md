# Tomo-NeRF

Tomo-NeRF is a tool that allows you to perform ultra-sparse view tomographic reconstruction. This repository provides the necessary files for training, optimization, and testing of the Tomo-NeRF model.

### Usage: 

#### Data Preparation: 
Tomo-NeRF requires a set of 2D radiographic projection images. Ensure that you have a set of 2-6 images, evenly spaced from 0 degrees to 180 degrees.  Here we offers a set of real x-ray imaging and artificial image samples in compressed files (unzip before use).

#### Training
Specifiy DATAPATH with the path to the directory containing your projection images. Adjust the parameter as requested in script based on the number of images/types of images you have.
The training script will save the trained model for each epoch. Please note here we only provide part of training samples due to the limitation of files sizes. You can access the trained model through the following link: [Trained Model](https://drive.google.com/drive/folders/1-uTtm3OzJTFJs3P851HDodRUjEA_llNw?usp=sharing)
