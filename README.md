# MaskNotify-Personalized-Alerts-for-Unmasked-Individuals

Using Python, OpenCV, TensorFlow, and Keras, our system detects masks on faces and alerts individuals without masks from a face database, ensuring safety. It also tracks mask movement for surveillance purposes.

![Picture1](https://github.com/University-projetcs/MaskNotify-Personalized-Alerts-for-Unmasked-Individuals/assets/104216615/6dee29bd-0555-41d3-9463-5444505bf4d0)

In this project, we propose a method in which a bounding box is drawn over a person's face to indicate whether or not the person is wearing a mask. Image detection is carried out using the HAAR-CASCADE algorithm. Together with other existing algorithms, this classifier can produce a high recognition rate, efficient feature selection, and low false-positive rates. An 85-95% recognition rate is obtained with HAAR's feature-based cascade classifier utilizing only 200 features of 6000 features. This is object detection and classification problem with two classes (with and without mask). A dataset is used to build this face mask detector using Python, OpenCV, TensorFlow, and Keras. Using the database of a person's face, it detects the name of the person not wearing a mask and warns them through speakers that they do not have a mask on so that they can take precautions. In addition to detecting a face, it can also detect the movement of a mask while performing surveillance tasks.

Here is the short demo of the project:
![video](https://github.com/University-projetcs/MaskNotify-Personalized-Alerts-for-Unmasked-Individuals/assets/104216615/11f24eb6-a1af-48f9-9de5-74a938376126)
