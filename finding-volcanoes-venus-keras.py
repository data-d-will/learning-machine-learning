import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras import layers
from keras import models
from keras import metrics
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
import time


# add ../ for
X_train = pd.read_csv( '../volcanoesvenus/Volcanoes_train/Volcanoes_train/train_images.csv', header=None)
y_train  = pd.read_csv('../volcanoesvenus/Volcanoes_train/Volcanoes_train/train_labels.csv')

# Load test data
X_test  = pd.read_csv('../volcanoesvenus/Volcanoes_test/Volcanoes_test/test_images.csv', header=None)
y_test  = pd.read_csv('../volcanoesvenus/Volcanoes_test/Volcanoes_test/test_labels.csv')

# 1K volcanes and 6K non-volcanoes
# print(len(y_train) - y_train.count())
# print(y_train.astype(bool).sum(axis=0)) 

# Normalize data
# take out only type of being a volcano
Xtrain_raw = X_train/255
ytrain_raw = y_train['Volcano?']
Xtest_raw = X_test/255
ytest_raw = y_test['Volcano?']

img_rows, img_cols = 110, 110

# reshape all records by the stated size
X = Xtrain_raw.values.reshape((-1, img_rows, img_cols, 1))
y = ytrain_raw.values
# print(y)
X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size = 0.2, random_state = 3)

X_test = Xtest_raw.values.reshape((-1, img_rows, img_cols, 1))
y_test = ytest_raw.values

#
# Function for plotting history of models
# take in  model history and path to write png of plot
#
def plot_history(model, path):
    acc = model.history['acc']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, model.history['acc'], color='#6b5b95',marker='o',linestyle='none',label='Training Accuracy')
    plt.plot(epochs, model.history['val_acc'], color='#d64161',label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs, model.history['loss'], color='#feb236', marker='o',linestyle='none',label='Training Loss')
    plt.plot(epochs, model.history['val_loss'], color='#ff7b25',label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig(path) 
    plt.close()


# X_train = tf.cast(X_train, tf.int32)
batch_size = 64

# baseline
# maxpool
model_maxpool = models.Sequential()
model_maxpool.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_maxpool.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_maxpool.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_maxpool.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_maxpool.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_maxpool.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_maxpool.add(layers.Flatten())
model_maxpool.add(layers.Dense(1, activation = 'sigmoid'))


model_maxpool.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
start_time = time.time()

epochs = 32


model_maxpool_history = model_maxpool.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size,
    validation_data=(X_vali, y_vali),
    shuffle=False
)

print('fitting time',time.time() - start_time)

plot_history(model_maxpool_history, '../plots/maxpool.png')

model_base_score = model_maxpool.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - baseline')
print(' - - - ')

#
# AVGPOOL
# epochs - 26 
model_avgpool = models.Sequential()
model_avgpool.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_avgpool.add(layers.AveragePooling2D(pool_size=(2,2), strides=2))
model_avgpool.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_avgpool.add(layers.AveragePooling2D(pool_size=(2,2), strides=2))
model_avgpool.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_avgpool.add(layers.AveragePooling2D(pool_size=(2,2), strides=2))
model_avgpool.add(layers.Flatten())
model_avgpool.add(layers.Dense(1, activation = 'sigmoid'))

model_avgpool.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 26
start_time = time.time()
model_avgpool_history = model_avgpool.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali),
    shuffle=False
)
print('fitting time',time.time() - start_time)
plot_history(model_avgpool_history, '../plots/avgpool.png')

model_base_score = model_avgpool.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - avgpool')
print(' - - - ')

#simplify baseline
# remove maxpool - no good epoch number
model_base_simple = models.Sequential()
model_base_simple.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_base_simple.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_base_simple.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_base_simple.add(layers.Flatten())
model_base_simple.add(layers.Dense(1, activation = 'sigmoid'))

model_base_simple.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 32
start_time = time.time()
model_base_simple_history = model_base_simple.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
)
print('fitting time',time.time() - start_time)

plot_history(model_base_simple_history, '../plots/simple.png')

model_base_score = model_base_simple.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - simple baseline')
print(' - - - ')

#
# kernel 3,1 & 1,3
# epochs 19 - 

model_3_1 = models.Sequential()
model_3_1.add(layers.Conv2D(9, kernel_size = (3,1), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_3_1.add(layers.Conv2D(9, kernel_size = (1,3), activation = 'relu'))
model_3_1.add(layers.MaxPool2D(pool_size=(3,3), strides=2))
model_3_1.add(layers.Conv2D(18, kernel_size = (3,1), activation = 'relu'))
model_3_1.add(layers.Conv2D(18, kernel_size = (1,3), activation = 'relu'))
model_3_1.add(layers.MaxPool2D(pool_size=(3,3), strides=2))
model_3_1.add(layers.Conv2D(36, kernel_size = (3,1), activation = 'relu'))
model_3_1.add(layers.Conv2D(36, kernel_size = (1,3), activation = 'relu'))
model_3_1.add(layers.MaxPool2D(pool_size=(3,3), strides=2))
model_3_1.add(layers.Flatten())
model_3_1.add(layers.Dense(1, activation = 'sigmoid'))

model_3_1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 19
start_time = time.time()
model_3_1_history = model_3_1.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
)
print('fitting time',time.time() - start_time)

plot_history(model_3_1_history, '../plots/pool_3_1.png')

model_base_score = model_3_1.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - 3_1')
print(' - - - ')


# kernel 3,1 & 1,3
# epochs - 19
model_1_3 = models.Sequential()
model_1_3.add(layers.Conv2D(9, kernel_size = (1,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_1_3.add(layers.Conv2D(9, kernel_size = (3,1), activation = 'relu'))
model_1_3.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_1_3.add(layers.Conv2D(18, kernel_size = (1,3), activation = 'relu'))
model_1_3.add(layers.Conv2D(18, kernel_size = (3,1), activation = 'relu'))
model_1_3.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_1_3.add(layers.Conv2D(36, kernel_size = (1,3), activation = 'relu'))
model_1_3.add(layers.Conv2D(36, kernel_size = (3,1), activation = 'relu'))
model_1_3.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_1_3.add(layers.Flatten())
model_1_3.add(layers.Dense(1, activation = 'sigmoid'))

model_1_3.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 19
start_time = time.time()
model_1_3_history = model_1_3.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
) 
print('fitting time',time.time() - start_time)

plot_history(model_1_3_history, '../plots/pool_1_3.png')

model_base_score = model_1_3.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - 1_3')
print(' - - - ')

# small dropout - good but more epochs and with decay
#
model_drop = models.Sequential()
model_drop.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_drop.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_drop.add(layers.Dropout(rate=0.1)) 
model_drop.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_drop.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_drop.add(layers.Dropout(rate=0.1)) 
model_drop.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_drop.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_drop.add(layers.Flatten())
model_drop.add(layers.Dropout(rate=0.2)) 
model_drop.add(layers.Dense(1, activation = 'sigmoid'))

model_drop.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 30
start_time = time.time()
model_drop_history = model_drop.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
) 
print('fitting time',time.time() - start_time)

plot_history(model_drop_history, '../plots/dropout.png')

model_base_score = model_drop.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - dropout')             
print(' - - - ')

#
# 2nd to final dense
# 
model_dense = models.Sequential()
model_dense.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_dense.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_dense.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_dense.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_dense.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_dense.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_dense.add(layers.Flatten())
model_dense.add(layers.Dense(36, activation='relu')) # 72: bad, 144: 0.965, 36:  0.963
model_dense.add(layers.Dense(1, activation = 'sigmoid'))

model_dense.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 22
start_time = time.time()
model_dense_history = model_dense.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
) 
print('fitting time',time.time() - start_time)

plot_history(model_dense_history, '../plots/Dense_36.png')


model_base_score = model_dense.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - small final dense')
print(' - - - ')

# 2,2 
#  
model_2_2 = models.Sequential()
model_2_2.add(layers.Conv2D(13, kernel_size = (2,2), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_2_2.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_2_2.add(layers.Conv2D(26, kernel_size = (2,2), activation = 'relu'))
model_2_2.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_2_2.add(layers.Conv2D(52, kernel_size = (2,2), activation = 'relu'))
model_2_2.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_2_2.add(layers.Flatten())
model_2_2.add(layers.Dense(1, activation = 'sigmoid'))

model_2_2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 36
start_time = time.time()
model_2_2_history = model_2_2.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
) 
print('fitting time',time.time() - start_time)
plot_history(model_2_2_history, '../plots/2_2.png')

model_base_score = model_2_2.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - 2_2')
print(' - - - ')

# 4,4
# 
# works pretty good
# need some dropout
model_4_4 = models.Sequential()
model_4_4.add(layers.Conv2D(13, kernel_size = (4,4), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_4_4.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_4_4.add(layers.Conv2D(27, kernel_size = (4,4), activation = 'relu'))
model_4_4.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_4_4.add(layers.Flatten())
model_4_4.add(layers.Dense(1, activation = 'sigmoid'))

model_4_4.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 36
start_time = time.time()
model_4_4_history = model_4_4.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
) 
print('fitting time',time.time() - start_time)

plot_history(model_4_4_history, '../plots/4_4.png')


model_base_score = model_4_4.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - 4_4')
print(' - - - ')

# 5,5
#
model_5_5 = models.Sequential()
model_5_5.add(layers.Conv2D(11, kernel_size = (5,5), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_5_5.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_5_5.add(layers.Conv2D(22, kernel_size = (5,5), activation = 'relu'))
model_5_5.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_5_5.add(layers.Flatten())
model_5_5.add(layers.Dense(1, activation = 'sigmoid'))

model_5_5.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 24
start_time = time.time()
model_5_5_history = model_5_5.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
) 
print('fitting time',time.time() - start_time)

plot_history(model_5_5_history, '../plots/5_5.png')


model_base_score = model_5_5.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - 5_5')
print(' - - - ')


#
# regularizer
# l2
model_kernel_l2 = models.Sequential()
model_kernel_l2.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_kernel_l2.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_kernel_l2.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_kernel_l2.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_kernel_l2.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_kernel_l2.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_kernel_l2.add(layers.Flatten())
model_kernel_l2.add(layers.Dense(144, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model_kernel_l2.add(layers.Dropout(rate=0.5)) 
model_kernel_l2.add(layers.Dense(1, activation = 'sigmoid'))

model_kernel_l2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 30
start_time = time.time()
model_kernel_l2_history = model_kernel_l2.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False
) 
print('fitting time',time.time() - start_time)

plot_history(model_kernel_l2_history, '../plots/reg-l2.png')

model_base_score = model_kernel_l2.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - reg_l2')
print(' - - - ')

# l1
model_kernel_l1 = models.Sequential()
model_kernel_l1.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_kernel_l1.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_kernel_l1.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_kernel_l1.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_kernel_l1.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_kernel_l1.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_kernel_l1.add(layers.Flatten())
model_kernel_l1.add(layers.Dense(144, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
model_kernel_l1.add(layers.Dropout(rate=0.5)) 
model_kernel_l1.add(layers.Dense(1, activation = 'sigmoid'))

model_kernel_l1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 30
start_time = time.time()
model_kernel_l1_history = model_kernel_l1.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    shuffle=False 
) 
print('fitting time',time.time() - start_time)

plot_history(model_kernel_l1_history, '../plots/reg-l1.png')

model_base_score = model_kernel_l1.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - reg_l1')
print(' - - - ')

# weighting
#
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# print(class_weight)

model_weight = models.Sequential()
model_weight.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_weight.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_weight.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_weight.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_weight.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_weight.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_weight.add(layers.Flatten())
model_weight.add(layers.Dense(1, activation = 'sigmoid'))

model_weight.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 32

start_time = time.time()
model_weight_history = model_weight.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_vali, y_vali), 
    class_weight=class_weight,
    shuffle=False
) 
print('fitting time',time.time() - start_time)

plot_history(model_weight_history, '../plots/weights.png')

model_base_score = model_weight.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - weights')
print(' - - - ')

# data augmentation
#
# first and selected augmentation model
#
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = False, 
    width_shift_range= False,  
    height_shift_range= False,  
    horizontal_flip=True,  
    vertical_flip=True)  

# second augmentation and not selected model
# 
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

datagen.fit(X_train)

stepsize = len(X_train) / batch_size
generator = datagen.flow(X_train, y_train, batch_size= batch_size)

# baseline model with data augmentation
model_aug = models.Sequential()
model_aug.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_aug.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_aug.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_aug.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_aug.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_aug.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_aug.add(layers.Flatten())
model_aug.add(layers.Dense(1, activation = 'sigmoid'))

model_aug.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
epochs = 60

start_time = time.time()
model_aug_history = model_aug.fit_generator(generator,epochs = epochs, steps_per_epoch=stepsize, validation_data = (X_vali,y_vali))
print('fitting time',time.time() - start_time)

plot_history(model_aug_history, '../plots/generator-aug.png')

model_base_score = model_aug.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - data aug')
print(' - - - ')


# trying to combine techniques to achieve even better results
model_aug_opti = models.Sequential()
model_aug_opti.add(layers.Conv2D(9, kernel_size = (3,3), activation = 'relu', input_shape = (img_rows, img_cols, 1)))
model_aug_opti.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_aug_opti.add(layers.Dropout(rate=0.1)) 
model_aug_opti.add(layers.Conv2D(18, kernel_size = (3,3), activation = 'relu'))
model_aug_opti.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_aug_opti.add(layers.Dropout(rate=0.1)) 
model_aug_opti.add(layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'))
model_aug_opti.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
model_aug_opti.add(layers.Dropout(rate=0.1)) 
model_aug_opti.add(layers.Dense(144, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model_aug_opti.add(layers.Dropout(rate=0.5))
model_aug_opti.add(layers.Flatten())
model_aug_opti.add(layers.Dense(1, activation = 'sigmoid'))

# 
epochs = 32
# optimizers.Nadam(lr=0.0022, schedule_decay=0.006)


model_aug_opti.compile(
    loss = 'binary_crossentropy', 
    # optimizer=optimizers.Nadam(), 
    optimizer=optimizers.Adam(), 
    metrics = ['accuracy']
)


start_time = time.time()
model_aug_opti_history = model_aug_opti.fit_generator(
    generator,
    epochs = epochs, 
    steps_per_epoch=stepsize, 
    validation_data = (X_vali,y_vali), 
    shuffle=False
)
print('fitting time',time.time() - start_time)

plot_history(model_aug_opti_history, '../plots/generator-aug-opti-adam-30.png')

model_base_score = model_aug_opti.evaluate(X_test, y_test)
print('Test Loss =', model_base_score[0])
print('Test Accuracy =', model_base_score[1])
print('model - combining techniques')

