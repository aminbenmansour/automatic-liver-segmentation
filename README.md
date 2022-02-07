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

# Stack

Using frameworks is fundamental. using the right frameworks for the job could immensely simplify the job.

For this project we used the following technologies.

* [MONAI](https://monai.io/): Medical Open Networkfor Artificial Intelligence, provides domain-optimized foundational capabilities for developing healthcare imaging training workflows in a native PyTorch paradigm. Useful for image pre-processing.

  <img width="20%" height="auto" style="float: left;" src="https://user-images.githubusercontent.com/50111205/152818445-364cbd51-e638-40f5-b3fb-8171d8ffd936.png" alt= "MONAI-logo-color">



* [PyTorch](https://pytorch.org/): An open source machine learning framework that accelerates the path from research prototyping to production deployment. It takes care of the rest of the life hooks of the model.

  <img width="20%" height="auto" style="float: left;" src="https://user-images.githubusercontent.com/50111205/152818452-8214c746-9c8d-4266-9323-bab7877ea116.png" alt= "Pytorch_logo">

# Software
[3D Slicer](https://www.slicer.org/): solve advanced image computing challenges with a focus on clinical and biomedical applications.

The software was extensively used during the data preparation hook. it is useful for **converting NifTI files to DICOM** and **labeling (manually segmenting) the non-annotated data**.

  <img width="20%" height="auto" style="float: left;" src="https://user-images.githubusercontent.com/50111205/152821465-41f497ad-2407-429b-9f7f-ce5f71ac2788.svg" alt= "3D-Slicer-Mark">

# Usage
```
pip install -r requirements.txt
```

# Datasets

[Decathlon dataset](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2), *labeled*

[Liver Tumor Segmentation - Part 1](https://www.kaggle.com/andrewmvd/liver-tumor-segmentation), *unlabeled*

[Liver Tumor Segmentation - Part 2](https://www.kaggle.com/andrewmvd/liver-tumor-segmentation-part-2), *unlabeled*

