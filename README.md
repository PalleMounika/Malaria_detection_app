# Malaria_detection_app
This repository was created to help clarify how to classify parasitized or not using malaria_detection_app.
A streamlit app running on the browser calls the interface where we upload a image to get classified.

This app is currently live and we can easily detect the malaria infected cells or not by using this app.

Here we first run cnn.py file which will "generate malaria-cnn-v1.keras" file which is pretrained with the data we had already provided. 
Then in hotspotdetection.py file we need to give path of the keras file which got generated with cnn.py file and also cascade.xml file. 
After that when we run hotspotdetection.py file through stream lit app, it will generate a local interface in which we can browse cell images and the app will
get classify the image as Parasitized or Uninfected and also it will highlight with rectangular boxes the region which got parasitized.

Customization options
Use your own model
Place your trained keras deep learning model to the models directory.

**Use other pre-trained model**
See Keras applications for more available models such as DenseNet, MobilNet, NASNet, etc.
