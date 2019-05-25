import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

base_dir = "../input/"
train_dir = os.path.join(base_dir,"train/train")
testing_dir = os.path.join(base_dir, "test")

train = pd.read_csv("../input/train.csv")
train_dataframe = pd.read_csv("../input/train.csv")
train_dataframe["has_cactus"] = np.where(train_dataframe["has_cactus"] == 1, "yes", "no")


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


classifier = Sequential()

# first convolution layer
classifier.add(Conv2D(filters=16, kernel_size=(3, 3),
                             input_shape = (32, 32, 3), activation="relu"))

# Max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# second convolution layer
classifier.add(Conv2D(32, kernel_size=(3, 3),
                             activation="relu"))

# Max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatteing layer
classifier.add(Flatten())

# Fully connected Layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   validation_split=0.25,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_dataframe(dataframe = train_dataframe,
                                                directory = train_dir,
                                                x_col="id",
                                                y_col="has_cactus",
                                                target_size=(32,32),
                                                subset="training",
                                                batch_size=25,
                                                shuffle=True,
                                                class_mode="binary")

val_set = train_datagen.flow_from_dataframe(dataframe = train_dataframe,
                                                    directory = train_dir,
                                                    x_col="id",
                                                    y_col="has_cactus",
                                                    target_size=(32,32),
                                                    subset="validation",
                                                    batch_size=25,
                                                    shuffle=True,
                                                    class_mode="binary")

classifier.fit_generator(training_set,
                         epochs = 100,
                         steps_per_epoch = 525,
                         validation_data = val_set,
                         validation_steps = 175)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    testing_dir,
    target_size=(32,32),
    batch_size=1,
    shuffle=False,
    class_mode=None
)

preds = classifier.predict_generator(
    test_generator,
    steps=len(test_generator.filenames)
)

image_ids = [name.split('/')[-1] for name in test_generator.filenames]
preds = preds.flatten()
data = {'id': image_ids, 'has_cactus':preds} 
submission = pd.DataFrame(data)
print(submission.head())

submission.to_csv("submissions.csv", index=False)