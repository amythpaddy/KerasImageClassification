from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense


classifier = Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(42, 42, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

classifier.add(Dense(output_dim = 128 , activation='relu'))
classifier.add(Dense(output_dim = 1 , activation='softmax'))

classifier.compile(optimizer= 'adam' , loss= 'binary_crossentropy' , metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    'images/train',
    target_size=(42,42),
    batch_size=32,
    class_mode='binary')
test_set= test_datagen.flow_from_directory(
    'images/test',
    target_size=(42,42),
    batch_size=32,
    class_mode='binary')

from IPython.display import display
from PIL import Image

classifier.fit_generator(
    training_set,
    steps_per_epoch=20,
    epochs=1,
    validation_data=test_set,
    validation_steps=800
)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('images/test_five.png', target_size=(42,42))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if(result[0][0] >= 0.5):
    prediction = 'eight'
else:
    prediction= 'five'

print(prediction + '---'+ result[0][0])