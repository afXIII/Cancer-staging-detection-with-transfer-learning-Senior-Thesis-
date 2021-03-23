### Ali Farahmand
### CS 488 Senior Capstone


#imports
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#commonly used variables
batch_size = 10
image_size = (96,96)
epochs = 10

#loading csv into dataframe
labels = pd.read_csv("histopathologic-cancer-detection/train_labels.csv")

#Creating the training and valiation sets
labels_neg = labels[labels['label'] == 0].sample(80000, random_state=3)
labels_pos = labels[labels['label'] == 1].sample(80000, random_state=3)

val_neg = labels_neg[:8000]
val_pos = labels_pos[:8000]

labels_neg = pd.concat([labels_neg, val_neg]).drop_duplicates(keep=False)
labels_pos = pd.concat([labels_pos, val_pos]).drop_duplicates(keep=False)


train_labels = pd.concat([labels_neg, labels_pos])
val_labels = pd.concat([val_neg, val_pos])

#adding image format .tif to the end of each id
def append_ext(fn):
    return fn+".tif"
train_labels["id"] = train_labels["id"].apply(append_ext)
val_labels["id"] = val_labels["id"].apply(append_ext)

#datagenerators and image augmentation
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
									width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
									horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(dataframe=train_labels, directory="histopathologic-cancer-detection/train", x_col="id", y_col="label", class_mode="raw", target_size=image_size, batch_size=batch_size)
val_generator = val_datagen.flow_from_dataframe(dataframe=val_labels, directory="histopathologic-cancer-detection/train", x_col="id", y_col="label", class_mode="raw", target_size=image_size, batch_size=batch_size)

#downloading the imagenet pre trained model on VGG16
vgg_model = VGG16(include_top = False,
                    input_shape = (96,96,3),
                    weights = 'imagenet')

# Freeze the layers 
for layer in vgg_model.layers[:-7]:
    layer.trainable = False

#Creating the model
model = Sequential()

model.add(vgg_model)
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

#saving the model with best val accuracy
filepath = "modelvgg.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

#reduce learning rate if val accuracy drops
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

#Training
history = model.fit(train_generator, batch_size = batch_size, epochs=epochs,
                              validation_data=val_generator, validation_steps=50, 
                              verbose=1,
                              callbacks = callbacks_list)

#graphing the accuracy and loss values
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Transfer Learning Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.savefig('plotvggfrozen.png')
