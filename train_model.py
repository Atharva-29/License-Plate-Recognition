from keras.preprocessing.image import ImageDataGenerator as idg
from keras.models import Sequential as seq
from keras.layers import Conv2D
from keras.layers import MaxPooling2D as MaxP2D
from keras.layers import Dense, Flatten, Dropout
import os

image_width, image_height = 128, 128
rescl = 1. / 255

training_dir = 'alpha_chfinal/training'
testing_dir= 'alpha_chfinal/testing'
training_samples = 136194
testing_samples = 21602

epoch = 15
batch_s= 200

model = seq()
model.add(Conv2D(32,[3,3],strides=1,padding='valid',activation='relu',
                        input_shape=(image_width, image_height, 3)))
model.add(Conv2D(32,[3,3],strides=1,padding='valid',activation='relu'))
model.add(MaxP2D(pool_size=(2,2)))

model.add(Conv2D(64,[3,3],strides=1,padding='valid',activation='relu'))
model.add(MaxP2D(pool_size=(2,2)))

model.add(Conv2D(64,[3,3],strides=1,padding='valid',activation='relu'))
model.add(Conv2D(128,[3,3],strides=1,padding='valid',activation='relu'))
model.add(MaxP2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(36, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = idg(
    rescale=rescl,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=5,
    horizontal_flip=True)

test_datagen = idg(rescale=rescl)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(image_width, image_height),
    batch_size=batch_s,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    testing_dir,
    target_size=(image_width, image_height),
    batch_size=batch_s,
    class_mode='categorical')

model.summary()
model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples//batch_s,
    epochs=epoch,
    validation_data=test_generator,
    validation_steps=testing_samples//batch_s)

model.save_weights('trfinal.h5')