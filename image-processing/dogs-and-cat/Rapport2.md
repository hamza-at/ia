


**Master 2 Logiciels sur:** 

Artificial intelligence






**Cats and Dogs classification**






Professor:

`	`-**Mme Delphine MAUGARS** 



Group :

\- **AIT MANSOUR  HAMZA**

**- AMROUNI MASSINISSA**










---------------------------------------------------  **Plan** --------------------------------------------------------

**Implementation choice**

**Initial model**

**changement of parameters**

**The best model**

**-------------------------------------------------------------------------------------------------------**
























**Implementation choice**

** We checked that codes are running very fine into Kaggle python notebook but same is not running correctly into my local machine. Thats why we used Kaggle and uploaded some random Cat and dog dataset.

Our code is based on TesnorFlow and Keras, thats allow us to observe models performances  and modify paramaters if necessary so that we achieve the best precision score possible.

Our implentation execute the folloxing tasks:

` `unzip cat and dog files(train and test) and prepare them for processing.

` `visualize some of data to make sure of size and content .

` `Creation of models and train them.

` `Testing precision of our model.











`        `**Initial model**						

**To begin, here are the different parameters of our first cnn classification model dog**

**and cat :**

`   `**number of epochs: 20**

`   `**number of layer(all layers): 17**

**Here are the 17 layers used :**

`   `**Conv2D avec with activation "relu" and a kernel\_size (3,3)**           

`    `**and 64 neurons and also with 128 neurons**

`   `**Maxpool2D**

`    `**Dropout a 0.4**

`     `**Flatten**

`     `**Dense activation "relu" (256 neurons)** 

`     `**0.0005 as learning rate and 1e -5 as decay\_rate**

`    `**Optimizer= adam** 

**For the number images used in training, validation and test , we have the following :**

`      `**15000 images for training,  5000 for validation and the 	rest for the testing phase** 

`   `**Results**

`   `**Training\_loss: 0.1191**

`   `**Training\_acc: 0.9668**

`   `**Validation\_loss: 0.1493**

`   `**Validation\_acc: 0.9496**

**Good results so far, but not satisfying**





**changement of parameters**

**we tried to change the parameters to increase model precision and effectiveness. So we selected the following parameters** 

`    `**number of epochs : training the model a little bit more than te initial model**

`    `**number of layers : the features extracted will be more precise and specific**

`   `**Learning rate ( = learning  speed. By decelarate the speed we may give the model more time to extract more features )**

`  `**Régularisation (Dropput layers) = prevents cnn from overfitting**

`  `**Optimizer (we used Adam, but we just found out about RMSProp and we want to give it a try)**

LET'S START WITH THE NUMBER OF EPOCHS

`  `**Nombre d’epochs**

**15 epochs :**

` `**Training\_loss: 0.0886** 

` `**Training\_acc: 0.9744** 

` `**Validation\_loss: 0.1857** 

` `**Validation\_acc: 0.9598**




**20 epochs :**

`  `**Training\_loss: 0.6721**

` `**Training\_acc: 0.9824** 

` `**Validation\_loss: 0.1573**

` `**Validation\_acc: 0.9601**




**Increasing the number of epochs results the following:**

\- A low training\_loss 

\- Important validation and training accuracy

**Learning speed (learning rate et decay\_rate) , Optimizer**

**Learning rate :0.0005 , decay\_rate : 1e-5 , optimizer adam :**

`  `**Training\_loss: 0.6721**

` `**Training\_acc: 0.9824** 

` `**Validation\_loss: 0.1573**

` `**Validation\_acc: 0.9601**

**Learning rate :0.0001 , decay\_rate : 1e-5 , optimizer adam :**

`  `**Training\_loss: 0.0375**

`  `**Training\_acc: 0.9947** 

`  `**Validation\_loss: 0.0981**

`  `**Validation\_acc: 0.9637**

A decrease in the learning rate = low training loss

**Learning rate :0.001 , decay\_rate : 1e-6 , optimizer adam :**

`  `**Training\_loss: 0.2391**

`  `**Training\_acc: 0.9530**

`  `**Validation\_loss: 0.1849**

`  `**Validation\_acc: 0.9440**

When we go faster we lose informations and consequently the Training\_loss becomes important ( 0.2 compared to 0.03 with the previous rate).

**Learning rate :0.0005 , decay\_rate : 1e-5 , optimizer RMSprop :**

`  `**Training\_loss: 0.3043** 

`  `**Training\_acc: 0.8764**

`  `**Validation\_loss: 0 .4714**

`  `**Validation\_acc: 0.8445**

**Learning rate :0.0001 , decay\_rate : 1e-6 , optimizer RMSprop :**

`  `**Training\_loss: 0. 3402** 

`  `**Training\_acc: 0.8570** 

`  `**Validation\_loss: 0.3643**

`  `**Validation\_acc: 0.8597**


**the model seems to be more successful in learning than for the “Adam” optimizer + small learning rate** 

The smaller the loss, the better  the classifier is at modeling the relationship between the input data and the output targets

**number of layers , number of neurons and activation function**

We used also the following layers (Dense,Flatten, Conv2D and Maxpool2D), But in this study we gonna focus more on the other layers



**17 layers , number of neurons (64 x 2 , 128 x 3 ,256 x 3 ) , activation "relu" and regularisation of (0.4) , Conv2D with kernel\_size  (3,3) ,… :**

` `**Training\_loss: 0.0886** 

` `**Training\_acc: 0.9744** 

` `**Validation\_loss: 0.1857** 

` `**Validation\_acc: 0.9598**

**17 layers , number of neurons  (64 x 2 , 128 x 3 ,256 x 3 ) activation "relu" and regularisation of (0.2) , Conv2D with kernel\_size  (3,3) ,… :**

` `**Training\_loss: 0.0360** 

` `**Training\_acc: 0.9876** 

` `**Validation\_loss: 0.3235** 

` `**Validation\_acc: 0.9652**

**17 layers , number of neurons (64 x 2 , 128 x 3 ,256 x 3 ) , activation "relu" and regularisation of (0.7) , Conv2D with kernel\_size  (3,3) ,… :**

` `**Training\_loss: 29.4446**

` `**Training\_acc: 0.4936** 

` `**Validation\_loss: 0.6235** 

` `**Validation\_acc: 0.5000**

**modifying the regularization layer gives better results. Too high a dropout can slow down the convergence rate of the model and often affect the final performance. A rate that is too low provides little or no improvement in generalization performance.**

**12 layers , number of neurons (64 x 2 , 128 x 3 ) , utilisation activation "relu" and regularisation of (0.4) , Conv2D with kernel\_size  (3,3) ,… :**

` `**Training\_loss: 0.0343** 

` `**Training\_acc: 0.9841** 

` `**Validation\_loss: 0.4397**

` `**Validation\_acc: 0.9048**

**less layer makes the model less efficient**








**THE BEST MODEL**

**number of epochs 20**

**number of layers  17 :** 

`  `Conv2D 64 neurons and 128 neurons, activation "relu",            kernel\_size  (3,3) 

`  `Maxpool2D 

`  `Dropout(0.2)

`  `Flatten

`  `Dense activation "relu" (256 neurons) 

`  `0.0003 as learning rate and 1e-5 as decay\_rate

`  `Optimizer adam



Results:

` `**Training\_loss: 0.7345**

` `**Training\_acc: 0.9983** 

` `**Validation\_loss: 0.6981**

` `**Validation\_acc: 0.9674**




**we added the following code to generate the confusion matrix:**


**-----------------------------------------------------------------**

***from sklearn.metrics import confusion\_matrix***

***from PIL import Image***

***import matplotlib.pyplot as plt***

***import numpy as np***

***import torch***

***import torchvision***

***y\_true =[]***


***for file in os.listdir("./test1") :***

`    `***if file.startswith("cat"):***

`        `***y\_true.append(0)***

`    `***else :***

`        `***y\_true.append(1)***
\***

\***


***print(confusion\_matrix(y\_true, y\_pred\_classification))***

**--------------------------------------------------------------------------**

**we get the following matrix**


**             



