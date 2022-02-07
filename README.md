# automatic-liver-segmentation

# Introduction

Deep learning is an important machine learning technique that helps computers identify objects in images. Medical images are used in diagnosis and treatment of diseases, injuries, and congenital abnormalities.

Deep learning can be used to detect diseases like cancer by finding tumor cells through medical images. Deep learning also has the potential to improve the quality of medical care by segmenting organs during surgery or scanning patients for signs of cancer or other ailments.

The goal of this repository is to create a deep learning model to segment a liver from a public CT scan dataset. After completing this series, you will be able to create the same model that segments the liver, as well as use the same principle to segment other organs or tumors from CT scans or MRIs.

# Segmentation

Segmentation is one of various applications in computer vision. It does not only figure out the nature of the object (classifying) or draws a box around it (object detection), instead it adds a whole layer above the original DICOM file. both images has the same dimensions and the latter emphasizes the object we are working on.

The picture below shows the difference between the techniques mentioned above.

![1__MNNHEI2TjyX6R0dxGtsig](https://user-images.githubusercontent.com/50111205/152811779-bc681650-f473-46b0-bf75-61c9da7fd1c4.png)



# U-Net (Model)

The u-net is **Convolutional Networks for Biomedical Image Segmentation**. It is a convolutional neural network architecture that expanded with few changes in the CNN architecture. it was initially conceived for medical. It was invented to deal with biomedical images where the target is **not only to classify whether there is an infection or not but also to identify the area of infection.**. the U-Net has later been adopted in multiple fields thanks to it's effeciency.

The illustration below shows the architecture of the U-Net

<p align="center">
  <img src="https://user-images.githubusercontent.com/50111205/152804685-70b35b1b-f368-411f-a3a2-a79b3094108b.png" alt="Convolutional-neural-network-CNN-architecture-based-on-UNET-Ronneberger-et-al"/>
</p>

# Usage
```
pip install -r requirements.txt
```

# Datasets

[Decathlon dataset](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2), choosing Task 3, the liver

[Liver Tumor Segmentation - Part 1](https://www.kaggle.com/andrewmvd/liver-tumor-segmentation).

[Liver Tumor Segmentation - Part 2](https://www.kaggle.com/andrewmvd/liver-tumor-segmentation-part-2).


I personnaly tried the decathlon dataset it is recommended by the pytorch team but kaggle datasets are pretty interesting too.
