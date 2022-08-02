
# Image Caption Generator

Image Captioning is the process of generating a textual description for given images. It
has been a very important and fundamental task in the Deep Learning domain. Image
captioning has a huge amount of application. This work is an internship based project to develop a model for extracting image features and based on that, try to generate captions using neural networks.




## Dataset
Flickr8K Dataset is used which is easily available.
## Description
* Used Inceptionv3 model to extract image features.
* LSTM model to predict the next word of the captions sequence.



## File description
* Templates file contains the HTML files for the project.
* app.py contains the flask based application.
* Image.ipynb contains the notebook where the model is trained
* Model weights contains the trained models for caption generation
## Instructions
* Download the folder in your machine.
* Open cmd.exe in folder.
* Use command $ pip install -r requirements.txt.
* Use $ flask run
* Go to the local host link.


## Examples
<img src="https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?ixlib=rb1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8aHVtYW58ZW58MHx8MHx8&w=1000&q=80.png"  width="200"/><br>man is squatting on his back in the mountains

![image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS98AfLVtejR2Dz4IDrkKNtxZnlxaU_xIO5Xw&usqp=CAU)
<br>two people are walking on the side of road
