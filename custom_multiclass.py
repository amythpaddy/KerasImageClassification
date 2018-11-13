from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.utils.np_utils import  to_categorical

classifier = Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(42, 42, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

classifier.add(Dense(output_dim = 128 , activation='relu'))
classifier.add(Dense(output_dim = 2 , activation='sigmoid'))

classifier.compile(optimizer= 'adam' , loss= 'categorical_crossentropy' , metrics = ['accuracy'])

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
    class_mode='categorical')
test_set= test_datagen.flow_from_directory(
    'images/test',
    target_size=(42,42),
    batch_size=32,
    class_mode=None)

from IPython.display import display
from PIL import Image

train_labels = training_set.classes
train_labels = to_categorical(train_labels,num_classes=2)
classifier.fit(
    training_set,train_labels,
    epochs=1,
    batch_size=16,
    validation_data=(test_set,train_labels)
)

classifier.save_weights('my_custom_classifie.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('images/test_five.png', target_size=(42,42))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis=0)
result = classifier.predict_classes(image)
print(result[0])