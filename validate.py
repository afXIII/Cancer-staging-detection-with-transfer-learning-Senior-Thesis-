### Ali Farahmand
### CS 488 Senior Capstone

#imports
import pandas as pd
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

#commonly used variables
batch_size = 10
image_size = (96,96)

#load csv into dataframe
labels = pd.read_csv("histopathologic-cancer-detection/train_labels.csv")

#finding images used in training
labels_neg = labels[labels['label'] == 0].sample(80000, random_state=3)
labels_pos = labels[labels['label'] == 1].sample(80000, random_state=3)

#finding images not used in training
not_used_for_training = pd.concat([labels, labels_neg, labels_pos]).drop_duplicates(keep=False)

#creating the testing dataset
testing_neg = not_used_for_training[not_used_for_training['label'] == 0].sample(8000, random_state=3)
testing_pos = not_used_for_training[not_used_for_training['label'] == 1].sample(8000, random_state=3)

test_labels = pd.concat([testing_neg, testing_pos])

#adding .tif file format to image names
def append_ext(fn):
    return fn+".tif"
test_labels["id"] = test_labels["id"].apply(append_ext)

#datagenarator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(dataframe=test_labels, directory="histopathologic-cancer-detection/train", x_col="id", y_col="label", class_mode="raw", target_size=image_size, batch_size=batch_size)


#loading model for basic CNN
cnnModel = keras.models.load_model('model.h5')

#evaluating model for basic CNN
cnnModel.evaluate(test_generator)


print("Basic CNN Model evaluation Done!")

#loading model for VGG CNN with transfer learning
cnnModel = keras.models.load_model('modelvgg.h5')

#evaluating model for VGG CNN with transfer learning
cnnModel.evaluate(test_generator)

print("VGG CNN with transfer learning Model evaluation Done!")

#loading model for VGG CNN with transfer learning with less data
cnnModel = keras.models.load_model('modelvggless.h5')

#evaluating model for VGG CNN with transfer learning with less data
cnnModel.evaluate(test_generator)

print("VGG CNN with transfer learning Model with less data evaluation Done!")

