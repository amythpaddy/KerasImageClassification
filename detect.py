import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2


img_width, img_height = 42, 42
top_model_weights_path = 'my_custom_model.h5'
# top_model_weights_path = 'bottleneck_fc_model.h5'
training_data_dir='images/train'
validation_data_dir='images/test'

epochs = 1
batch_size=16


def predict():
    class_dictionary=np.load('class_indices.npy').item()
    num_classes = len(class_dictionary)
    image_path = 'images/test_eight.png'
    # image_path = 'images/test_five.png'
    orig = cv2.imread(image_path)

    print('[INFO] loading and processing Image...')
    image = load_img(image_path, target_size=(42,42))
    image = img_to_array(image)
    # image = image/255;
    image = np.expand_dims(image, axis=0)

    model = applications.VGG16(include_top=False, weights='imagenet')
    bottleneck_prediction = model.predict(image)

    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes , activation='sigmoid'))
    model.load_weights(top_model_weights_path)

    class_predicted = model.predict_classes(bottleneck_prediction)
    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]
    inv_map = {v: k for k,v in class_dictionary.items()}
    label = inv_map[inID]
    print("Image ID: {}, Label: {}".format(inID, label))

    # display the predictions with the image
    cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# save_bottleneck_features()
# train_top_model()
predict()
