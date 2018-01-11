# Part 1 Building CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import os

# Intializing CNN
classifier = Sequential()

# Convolution Layer
# 32 feature detector with 3 by 3 dimesnion
# input_size = (64,64,3) 64 by 64 image and 3 correspond to color .... (3,64,64) - Theano backend
classifier.add(Conv2D(32 , ( 3 , 3 ) , input_shape = (64,64,3) , activation= 'relu' ))
# Pooling - Using Max Pooling
classifier.add(MaxPooling2D( pool_size=(2,2)))

# adding a second convolution layer
classifier.add(Conv2D(32 , ( 3 , 3 ) , activation= 'relu' ))
classifier.add(MaxPooling2D( pool_size=(2,2)))

# Flattening - converting the data to vector of values
# which will be input to CNN
classifier.add(Flatten())

# Full connection - Fully conected layer( Hidden Layer)
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate = 0.1 ))

# Full connection - Fully conected layer( Hidden Layer)
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate = 0.1 ))



# Output layer (Soft max activation function for multiple result)
classifier.add(Dense(units=1, activation='sigmoid'))

# compiling the CNN (Scholatin radiant function (adam) , lost function(result type) and Performance metric )
classifier.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy'] )


#### Fit CNN to Image dataset ######
# Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=1,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=1,
                                            class_mode='binary')

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=18,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=18)

script_dir = os.path.dirname(__file__)

#### Predict a animal
import numpy as np
from keras.preprocessing import image



# Load Image
test_image = image.load_img('dataset/homework/cat.jpg' , target_size=(64,64)   )
# converting image to 3 dimension array
test_image = image.img_to_array(test_image)
# adding one more dimension as predict expect 4 dimension
# predict method wants input in batch
test_image = np.expand_dims(test_image , axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if(result[0][0] == 1) :
    prediction = 'dog' 
else:
    prediction = 'cat'
print(prediction)











