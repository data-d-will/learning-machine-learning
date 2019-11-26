import numpy as np
from PIL import Image

from keras import regularizers
from keras import layers
from keras import models
from keras import metrics
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

img_size = 80

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
    directory=r"flowers_dataset/flowers/train/",
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=20,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
# classes=["daisy", "dandelion", "rose", "sunflower", "tulip"],

valid_generator = test_datagen.flow_from_directory(
    directory=r"flowers_dataset/flowers/val/",
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=20,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=r"flowers_dataset/flowers/test/",
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

# First model
# 
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(5, activation='softmax'))


# Conv2D since learning about images
# Maxpooling to halve the input in both dim 
# AvgPool - Sum all of the values and dividing it by the total number of values 
# MaxPool - Selecting the maximum value
# Dropout is used to prevent overfitting
# last layer is softmax since it is good for multi-class - gives prob
#

model = models.Sequential()
model.add(layers.Conv2D(filters=20, kernel_size=(3,3), activation='relu', input_shape=(img_size,img_size,3))) 
# model.add(layers.Dropout(0.25)) 
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=40, kernel_size=(3,3), activation='relu'))
model.add(layers.Dropout(0.25)) 
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=80, kernel_size=(3,3), activation='relu'))
# model.add(layers.Dropout(0.25)) 
# model.add(layers.Conv2D(filters=80, kernel_size=(3,3), activation='relu'))
model.add(layers.Dropout(0.25)) 
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=160, kernel_size=(3,3), activation='relu'))
# model.add(layers.Conv2D(filters=160, kernel_size=(3,3), activation='relu'))
model.add(layers.Dropout(0.25)) 
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# model.add(layers.Conv2D(filters=320, kernel_size=(2,2), activation='relu'))
# model.add(layers.Conv2D(filters=320, kernel_size=(2,2), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.5)) 

model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(320, kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
# model.add(Dense(
#     256,
#     activation='relu'),
#     kernel_regularizer=regularizers.l2(0.01),
#     activity_regularizer=regularizers.l1(0.01)
# )
# model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
# activation='relu', input_shape=(10000,)))
model.add(layers.Dense(5, activation='softmax'))

#
# 
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# adam = optimizers.Adam(lr=0.005)

# high lr, <0.003, not good 
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.002, decay=1e-6), metrics=['accuracy'])
# next just prints more
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[metrics.categorical_crossentropy, metrics.categorical_accuracy])

# fitting
STEP_SIZE_TRAIN=len(train_generator) - 1
STEP_SIZE_VALID=len(valid_generator) - 1
STEP_SIZE_TEST=len(test_generator)

#36 epochs non aug
# 44 epochs aug
model_evo = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=44
    # use_multiprocessing=True,
    # workers=3
)
# model.summary()

# plotting  
acc = model_evo.history['acc']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
plt.plot(epochs, model_evo.history['acc'], color='#6b5b95',marker='o',linestyle='none',label='Training Accuracy')
plt.plot(epochs, model_evo.history['val_acc'], color='#d64161',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, model_evo.history['loss'], color='#feb236', marker='o',linestyle='none',label='Training Loss')
plt.plot(epochs, model_evo.history['val_loss'], color='#ff7b25',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')


plt.show()

# test_generator.reset()
#                                                                                       loss and acc 
# score = model.evaluate_generator(valid_generator, steps=STEP_SIZE_VALID, verbose=1)
# score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
# print('Start of evaluate generator')
# print('score:', score) # score: [1.107562252956643, 0.5295508351731808]
# print('End of evaluate generator')
# mean_loss = np.mean(loss)
# mean_acc = np.mean(acc)
# print('mean_loss:', mean_loss)
# print('mean_acc:', mean_acc)



# predict the output

print('predict generator')
test_generator.reset()
pred = model.predict_generator(
    test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1
)
# print('metrics')
# print(model.metrics)

#                                                                                           confusion matrix
pred_bool = (pred >0.5) # makes a different in confusion matrix
predictions = np.argmax(pred_bool, axis=1)
print('Confusion Matrix')
confMatrix = confusion_matrix(test_generator.classes, predictions)
print(confMatrix)
# plot_confusion_matrix(test_generator.classes, confMatrix, , figsize=(10,10))
# plot_confusion_matrix(test_generator.classes, predictions, normalize=True)


# print('Classification Report')
# target_names = ['daisy', 'dandelion', 'rose','sunflower','tulip']
# print(classification_report(test_generator.classes, predictions, target_names=target_names))
# print('test_generator', test_generator)