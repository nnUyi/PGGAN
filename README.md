# PGGAN
A implement of PGGAN for tensorflow version(progressive growing GANs for improved quality, stability and variation)
  
# Requirement
  
  - pytorch 0.2.0_4
  
  - python 2.7.12
  
  - numpy 1.13.1
  
  - scipy 0.17.0
  
# Usage
  ## download repo
  
    $ git clone https://github.com/nnUyi/PGGAN.git
    $ cd PGGAN
    
  ## download dataset
   - download [dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and store it in the datasets directory(directory named datasets) and then unzip it. Here I show the example of your dataset storage: /datasets/celebA. In this repo, celebA data is used to train the model, with no attributes and labels used in training time. 
  
  - I just use original celebA dataset, and target resolution is setted to 128*128 because I can not download the [delta data](https://drive.google.com/open?id=0B4qLcYyJmiz0TXY1NG02bzZVRGs) provided by NVIDIA.
  
  - Anyway, if your want to get 1024*1024 dataset, you can see [official codes](https://github.com/tkarras/progressive_growing_of_gans) here, h5tool.py is the script used to create the target training datasets.
  
  ## training
  
    $ python main.py --is_training=True 
    
  ## sampling
  
  - Sampling process is executed in training time. You can see the sampling results in the directory named sample

# Experiments
  The result shows below, we can clearly obtain such a good experimental result. Here I just show you two types rsolution including 64*64 and 128*128, the first four columns in 64*64 images are sampling data while the other four columns are real images. This is the same for 128*128 images.
  
  |sampling image|sampling image|
  |:-----------------:|:----------------:|
  |![Alt test](/data/64_64_1.png)|![Alt test](/data/64_64_2.png)|
  |64_64 resolution|64_64 resolution||
  |![Alt test](/data/128_128_1.png)|![Alt test](/data/128_128_2.png)|
  |64_64 resolution|64_64 resolution||
  
# Reference

  This repo is finished by referring to [github-pengge/PyTorch-progressive_growing_of_gans](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/README.md)
  
# Contacts
  
  Email:computerscienceyyz@163.com, Thank you for contacting if you find something wrong or if you have some problems!!!
