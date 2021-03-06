import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2

img_width, img_height = 42, 42
top_model_weights_path = 'my_custom_model.h5'
training_data_dir='images/train'
validation_data_dir='images/test'

epochs = 100
batch_size=16

def save_bottleneck_features():
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(42,42,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, 3, input_shape=(42, 42, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        training_data_dir,
        target_size=(img_width,img_width),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples/batch_size))

    bottleneck_feature_train= model.predict_generator(generator, predict_size_train);
    np.save('bottleneck_features_train.npy',bottleneck_feature_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    nb_validation_sample = len(generator.filenames)
    predict_size_validation = int(math.ceil(nb_validation_sample/batch_size))
    bottleneck_feature_validation = model.predict_generator(generator , predict_size_validation)
    np.save('bottleneck_features_validation.npy', bottleneck_feature_validation)

def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1./255)
    generator_top = datagen_top.flow_from_directory(
        training_data_dir,
        target_size=(img_width , img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    np.save('class_indices.npy',generator_top.class_indices)
    train_data = np.load('bottleneck_features_train.npy')
    train_labels=generator_top.classes
    train_labels=to_categorical(train_labels,num_classes=num_classes)

    generator_top=datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    nb_validation_sample=len(generator_top.filenames)
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels=generator_top.classes
    validation_labels=to_categorical(validation_labels,num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    history=model.fit(train_data,train_labels,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(validation_data,validation_labels))
    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(validation_data,
                                                validation_labels,
                                                batch_size=batch_size,
                                                verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    model.save("test.h5")
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

def predict():
    class_dictionary=np.load('class_indices.npy').item()
    num_classes = len(class_dictionary)
    image_path = 'images/test.JPG'
    # image_path = 'images/test_five.png'
    orig = cv2.imread(image_path)

    print('[INFO] loading and processing Image...')
    image = load_img(image_path, target_size=(42,42))
    image = img_to_array(image)
    image = image/255;
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

save_bottleneck_features()
train_top_model()
# predict()

cv2.destroyAllWindows()