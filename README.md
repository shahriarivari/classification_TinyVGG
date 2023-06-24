# image-classification-transformers
classifying cats and dogs images using TinyVGG architecture with Pytorch
using kagglecatsanddogs_3367a dataset
data_setup.py module chamges your data into DataLoader's object, also transform it and all
model_builder.py makes instances of TinyVGG architecture
model_trainer.py has the training and test steps
utils.py saves your model when training is done
and finally, train.py , in this module you can change the hyperparameters and includes the training and test loops.
