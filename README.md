# Image Segmentation on Massachusetts Roads Dataset : Approach

The world is everchanging, each day a new tree is planted, another one cut, a new road built and other one changed. It is difficult to manually note all the changes happening on the world terrain and this is where satellite imagery is used to understand the world better. However, the number of things (classes) to understand from satellite imagery is an arduous task manually. Intelligent computer systems using convolution neural networks (CNN) have been of great help to understand these. They have come a long way from identifying image in a picture (cat or dog) to identifying each pixel in the image as what it represents. The problem posed is an image segmentation problem where from the RGB satellite imagery, roads are required to be understood. The problem poses itself as a binary class problem – Road (1) or No-Road (0) for each pixel. The idea is to predict at pixel level as to which class the pixel can attributed.

**Files / Code**
1. EDA 
2. Training Pipeline
3. Prediction (Testing Batch) Pipeline 
4. Serving / Individual Inference

## Dataset

Dataset Link: https://www.kaggle.com/insaff/massachusetts-roads-dataset

The dataset is split into training (from where train and validation data is extracted) and testing directories. Each directory is further divided into two sub-directories 
1. input : holding input satellite imagery - files having 1500 x 1500 x 3 size : RGB kind 
2. output : holding the mask of the satellite imagery - files having 1500 x 1500 x 3 : Monochrome kind, yet with 3 channels

Observations
1. Training input has around 1105 files while training output has around 804 files, i.e. only 73% input images have corresponding ouput images.
2. Testing input and output has got 13 files each, however some output files (3 in number) are not aligned to input files
3. Number of black pixels in output files are far more than number of white pixels, as much as 19 times. It is an unbalanced dataset for models


### Dataset download

The dataset was downloaded from kaggle directly into personal google drive. It is possible to download the dataset to sample_data folder, however frequent google colab cut off means that everytime we have to download around 5GB of data.

### Dataset Processing (pre and post)

**TRAINING PIPELINE** : __Strategy_ : Images were not resized to model input size, instead the image was cropped._
1. Only the common files between training input and training output were used
2. Training and Validation data was divided in 0.85 : 0.15 
3. Tensorflow dataset object was used to create the data pipeline
4. Batch size was kept small for better learning
5. Image augmentation - cropping, flip vertical and horizontal was done (other augmentation were not done, as the domain knowledge was not there)
6. Image was cropped to 512x512x3 randomly and then flipped randomly
7. Only input image crops where the output image crop had more than 5% white pixels were used, this was done to help models train efficiently

**PREDICTION PIPELINE** : __Strategy_ : Image were broken into tiles (near to model input size), each tile was resized to model input size and predicted upon. The predicted file was than resized to original tile size and then all the tiles were joined together to reconstruct the ouput prediction
1. The predicted images were thresholded so that pixel brightness above 0.5 or 127 were made as 1 or 255, while anything below 0.5 was made 0.

**INFERENCE PIPELINE** : __Strategy_ : Similar to prediction pipeline

### Learnings
1. Resizing of images from original size to model size, does not help model training. Cropping delivered best result
2. Reconstructing tensorflow dataset on prediction pipeline is difficult due to large pre and post processing, regular python pipeline is used.
3. __Because of the low batch size, the compute spends more time in gathering the data rather than training the model. Tensorflow dataset pipeline provided good boost in this respect of reducing the data preparation (in-situ) time. Cache, repeat, prefetch were used as per tensorflow documentation.__

## Models

As there is a reconstrution of image involded - Encoder - Decoder type models were used.
Researching on image segmentation - Unet (U-Net: Convolutional Networks for Biomedical Image Segmentation : https://arxiv.org/abs/1505.04597) are used. U-net is a fully connected network (FCN) which first reduces the size of image (understand features better) and increases the size of the featues, with skip connection from encoder blocks to decoder blocks (so that learning is better and gradients are passed better)

Models used
1. U-net with MobileNetv2 (https://arxiv.org/abs/1801.04381v4) as encoder:
2. RESU-net (https://arxiv.org/pdf/1904.00592.pdf) with Residual Net with VGG 19 as encoder element

Changes from the original U-net to the employed models
1. The original U-net had different input and ouput sizes in width and height, the employed models had same input size and ouput size with respect to width and height
2. Before the skip connection were concatenated, they were cropped also, the employed models have sizes in power of 2, so no cropping is required
3. Pretrained weights on imagenets were used in encoder sections. This was because of compute issue. The smallest network - mobilenetv2 was used as encoder in one of the network while residual net was used in other - it seemingly has good results with aerial properties
4. The output layer has a sigmoid function with only 1 channel.

### Learnings
1. Custom U-net provided inferior results for same number of epochs viz a viz pretrained backbones
2. Dropout and Batchnormalization in Custom U-net helped increase training better, but were still of no match to pre-trained weight networks
3. The models were run for 50 epochs (due to compute restriction), but the trajectory of losses and other monitored metric indicate that if we train more there is a likelihood for better predictions
4. It may be not a bad idea for a model to overfit a bit. Underfitting is a problem.

## Model Training

### Training Hyperparameters
1. Split Ratio between test and validation was 0.85 : 0.15
2. Learning Rate was used as 1e-4, perhaps a learning rate finder could have been used
3. Epochs - 50, were used due to computring restrictions
4. Batch size of 4 was used. Low batch size helps model learn more in lesser epochs

### Custom Objects: Metrics and Loss
1. Custom metrics - Dice Score (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and Intersection over Union a.k.a (https://en.wikipedia.org/wiki/Jaccard_index). Existing metrics like Accuracy, Precision and Recall were also used.
2. Custom Loss - Dice Loss (1- Dice Score) was used to train the network, the reason for using this and not binary cross entropy loss was imbalanced dataset (black pixels outweight white pixels 19:1). Weighted binary cross entropy could also have been other option (not tried due to time)
3. Optimizer - Adam was used out of the box (based on literature review), with learning rate 1e-4
4. Callbacks - 
   1. Early Stopping : It was tried but not used later on as the validation loss was notoriously not moving much initially.
   2. Reduce Learning Rate on Plateau was used too
   3. Model Checkpoint was used to save the best model
   4. Backup Restore was used to start training after a crash (due to computing restrictions)

### Learnings
1. BackupRestore Callback was used to restart model training whenever compute used to crash. ModelCheckpoint was also used to provide the programming interrupt after x epochs and then restart the training. As after each interrupt the ram used to initialize 
2. Validation Loss was not decreasing - suggesting that the model is overfit, however the model did give good results on test. Given that the pretrained weights were used, it was not possible to  tinker with the model architecture to introduce regularizers like dropouts.

## Model Testing / Prediction
Model predictions for both models were availble, by changing the input param in the prediction pipeline.
The pipeline is written in .ipynb but .py file can be created as everything other than constants is encapsulated in functions

### Strategy
The image size of the satellite image is 1500 x 1500 x 3, making it a very big image. Ideal training of networks is done on image sizes which are in multiple of 2 i.e. 64,128,256,512,1024. However, it was seen that resizing the original image size to lower sizes, degrades the resolution a lot. Further training of the model was done using cropped images of size 512 x 512 x 3.

This leads us to the strategy where the original image was divided into tiles. 

1.   Original Image was divided into tiles : number of tiles = [ceil (original image size / model image size)]^2, which 9 tiles
2.   The image was equally divivded for 9 tiles giving each tile size of 500 x 500
3.   Each tile was then resize to 512 x 512 x 3 so that it can then become an input for the model
4.   Prediction was done for each time - giving us 9 ouputs of 512x 512 x 1. 
5.   Each predicted image was then resized to 500 x 500 x 1
6.   The predicted image was reconstructed using the predicted output

This strategy provided far better results viz a viz a pure resize and predict startegy

### Evaluation
1. Each test image predicted output was tested with the ground truth based on 4 metrics - IoU (Jaccard Score - one class), Accuracy, Precision and Recall
2. Each test image predicted output was visually inspected with ground truth. 3 images in ground truth were not aligned to input ones.

### Learnings
1. Resizing do not work much in prediction. Also evaluating images when the image size is reduced is not a good idea, because when the images are expanded there are artifacts which appear.
2. Tiling the image, predicting for each part and reconstructing the output image is the best way
3. Models have to be changed manually in the code - this can be automated later on


## Model Inference / Serving

Flask was used to serve the model running on Google Collab. Since the server address is not available a tunnel was created (using ngrok) to serve.
User can select a file and then see the prediction side by side.

### Learnings
1. The model selected (on which the inference is requried) has to be selected manually in code. this can be automated / understood from UI also 

### Future Work
1. Look at ways to increase metrics in lower compute
2. Better design model inferecing
3. Containerize the application


