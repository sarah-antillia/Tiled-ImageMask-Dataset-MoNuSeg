<h2>Tiled-ImageMask-Dataset-MoNuSeg (2024/07/08)</h2>

This is Tiled-ImageMask Dataset for MoNuSeg(Multi Organ Nuclei Segmentation) 2018.<br>
The dataset used here has been taken from the following web-site<br>
<b>Challenges/MoNuSeg/Data</b><br>
<pre>
https://monuseg.grand-challenge.org/Data/
</pre>
<!--
<b>Training Data</b><br>
The dataset (images and annotations) can be downloaded using the following links
<a href="https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view?usp=sharing"><b>MoNuSeg 2018 Training data</b></a><br>
<br>
<b>Testing Data</b><br>
Test set images with additional 7000 nuclear boundary annotations are available here
<a href="https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view?usp=sharing">
<b>MoNuSeg 2018 Testing</b></a>
-->
<br>

<b>Download Tiled-ImageMask-Dataset</b><br>
You can download our dataset from the google drive 
<a href="https://drive.google.com/file/d/1mUSsYRuISTS8bSzWXII_Hsc2PeA-39B6/view?usp=sharing">
Tiled-MoNuSeg-ImageMask-Dataset-M1.zip</a>
<br>


<h3>1. Dataset Citation</h3>
Please cite the following papers if you use the training and testing datasets of this challenge:<br>

N. Kumar et al., "A Multi-organ Nucleus Segmentation Challenge," in IEEE Transactions on <br>
Medical Imaging (in press) [Supplementary Information] [Code]<br>
N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, <br>
"A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology,"<br>
in IEEE Transactions on Medical Imaging, vol. 36, no. 7, pp. 1550-1560, July 2017 [Code]<br>
<br>

<b>License</b>: CC BY-NC-SA 4.0<br><br>

<h3>3. ImageMaskDataset Generation</h3>
Please download the master MoNuSeg Training and Testing data from the Google-drive..<br> 
<a href="https://drive.google.com/file/d/1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA/view?usp=sharing">
<b>MoNuSeg 2018 Training data</b></a><br>
<a href="https://drive.google.com/file/d/1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw/view?usp=sharing">
<b>MoNuSeg 2018 Testing</b></a>
<br>
<br>
Please run the following command for Python script <a href="./ImageMaskDatasetGenerator_1024x1024.py">
ImageMaskDatasetGenerator_1024x1024.py</a>.
<br>

This command generates two types of datasets from the master Training and Testing datatset, as shown below.<br>
<pre>
./MoNuSeg-master-1024x1024
├─images
└─masks
./mini_test
├─images
└─masks
</pre>
MoNuSeg-master-1024x1024 was generated from <b>MoNuSeg 2018 Training data</b>, and mini_test from 
<b>MoNuSeg 2018 Testing</b> respectively.<br>
<br>
<b>Training Data Tissue Images</b><br>
<img src="./asset/Train_data_tissue_images.png" width="1024" height="auto"></br>
<br>
<b>Testinng Data Tissue Images</b><br>
<img src="./asset/Test_data_tissue_images.png" width="1024" height="auto"></br>
<br>

As shown above, the number of the original Tissue-images in Training data is only 37, and too small to use for a training of a segmentation model.
Therefore, in order to increase of the number of the training data, we applied the following data augmentation methods to the original Training data.<br>
<li>hfip</li>
<li>vflip</li>
<li>rotation</li>
<li>shrinking</li>
<li>deformation</li>
<li>distortion</li>
<li>barrel_distortion</li>
<li>pincussion_distortion</li>
<br>
On detail of these methods, please refer to <a href="./ImageMaskDatasetGenerator_1024x1024.py">ImageMaskDatasetGenerator_1024x1024.py</a>.<br>
Furthermore, we resized all images and masks in the Training dataset 1024x1024 pixels from their original 1000x1000 pixels 
to create the MoNuSeg-master-1024x1024 dataset.
 In contrast, we retained all images and masks in the Testing dataset at their original 1000x1000 pixels to create the mini_test dataset
from the original Testing dataset.<br>
<br>


<h3>3. Tiled ImageMaskDataset Generation</h3>

Please run the following command for Python script <a href="./TiledImageMaskDatasetGenerator.py">
TiledImageMaskDatasetGenerator.py</a>.
<br>
<pre>
> python TiledImageMaskDatasetGenerator.py
</pre>
This command generates two types of images and masks:<br>
1. Tiledly-splitted to 512x512 image and mask files.<br>
2. Size-reduced to 512x512 image and mask files.<br>
Namely, this is a mixed set of Tiled and Non-Tiled ImageMask Datasets.<br>
<pre>
./Tiled-MoNuSeg-master
├─images
└─masks

</pre>
 
<h3>4. Split master</h3>

Pleser run the following command for Python <a href="./split_tiled_master.py">split_tiled_master.py</a> 
<br>
<pre>
>python split_tiled_master.py
</pre>
This splits Tiled-MoNuSeg-master into test, train and valid subdatasets.<br>
<pre>
./Tiled-MoNuSeg-ImageMask-Dataset-M1
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>
<hr>
Train images sample<br>
<img src="./asset/train_images_sample.png" width=1024 heigh="auto"><br><br>
Train mask sample<br>
<img src="./asset/train_masks_sample.png" width=1024 heigh="auto"><br><br>


Dataset Statistics <br>
<img src="./Tiled-MoNuSeg-2018-ImageMask-Dataset-M1_Statistics.png" width="512" height="auto"><br>
