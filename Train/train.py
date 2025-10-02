import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print("TensorFlow version:", tf.__version__)


import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import save_model, load_model
import math
import sys
from numpy import array,append,arange,zeros,exp,sin,random,std

import time

import numpy as np

path = 'path of dataset'

# Define the class types
train_types = ["type0", "typeIa", "typeIb", "typeIIa", "typeIIb", "typeIII", "typeIV", "typeV"]
val_types   = ["type0", "typeIa", "typeIb", "typeIIa", "typeIIb", "typeIII", "typeIV", "typeV"]

# Load training data
train_lc_list, train_shape_list = [], []
for t in train_types:
    train_lc_list.append(np.load(path+f"FinalData/LC/train_{t}.npy"))
    train_shape_list.append(np.load(path+f"FinalData/OM/train_{t}.npy"))

train_lc = np.concatenate(train_lc_list, axis=0)
train_shape = np.concatenate(train_shape_list, axis=0)

# Load validation data
vald_lc_list, vald_shape_list = [], []
for t in val_types:
    vald_lc_list.append(np.load(path+f"FinalData/LC/val_{t}.npy"))
    vald_shape_list.append(np.load(path+f"FinalData/OM/val_{t}.npy"))

vald_lc = np.concatenate(vald_lc_list, axis=0)
vald_shape = np.concatenate(vald_shape_list, axis=0)



train_dataset = tf.data.Dataset.from_tensor_slices((train_lc, train_shape))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.batch(100)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
print(train_dataset)

## val_d Set
vald_dataset = tf.data.Dataset.from_tensor_slices((vald_lc, vald_shape))
vald_dataset = vald_dataset.cache()
vald_dataset = vald_dataset.batch(100)
vald_dataset = vald_dataset.prefetch(tf.data.AUTOTUNE)

print(vald_dataset)

# CNN Model
input_shape = np.array(np.shape(train_lc[0]))
print("np.shape(input_shape) =", input_shape[0])

output_shape = np.array(np.shape(train_shape[0]))
print("np.shape(input_shape) =", output_shape[0], output_shape[1])

START = input_shape[0]
END = output_shape[0]
print("Start =", START)
print("End =", END)

conv_ip = keras.layers.Input(shape=(START,), name='Input')
x= keras.layers.Reshape((START, 1), input_shape=(START,), name='reshape_1')(conv_ip)
x= keras.layers.BatchNormalization()(x)

x=keras.layers.Conv1D(16,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='conv16_5',
                      padding='same')(x)

x=keras.layers.Conv1D(16,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='second_conv16_5',
                      padding='same')(x)

x=keras.layers.MaxPool1D(5,
                         strides=2,
                         data_format='channels_last',
                         name='maxpool_1',
                         padding='same')(x)

x=keras.layers.Conv1D(32,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='first_conv32_5',
                      padding='same')(x)

x=keras.layers.Conv1D(32,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='second_conv32_5',
                      padding='same')(x)

x=keras.layers.MaxPool1D(5,
                         strides=2,
                         data_format='channels_last',
                         name='maxpool_2',
                         padding='same')(x) #200

x=keras.layers.Conv1D(64,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='first_conv64_5',
                      padding='same')(x)

x=keras.layers.Conv1D(64,
                      kernel_size=5,
                      strides=1,
                      activation='relu',
                      name='second_conv64_5',
                      padding='same')(x)

x=keras.layers.MaxPool1D(5,
                         strides=2,
                         data_format='channels_last',
                         name='maxpool_3',
                         padding='same')(x)

x=keras.layers.Flatten(name='flat_1')(x)

x=keras.layers.Dense(256, name='dense_layer_5', activation='relu')(x)
x=keras.layers.Dense(256, name='dense_layer_6', activation='relu')(x)

x= keras.layers.Dense(END**2, name='dense_layer_u', activation='relu')(x)
x = keras.layers.Reshape(target_shape=(END, END, 1), name='reshape_2')(x)

x=keras.layers.Conv2D(32,
                       kernel_size=(3,3),
                       strides=1,
                       activation='relu',
                       name='second_conv64_52',
                       padding='same')(x)

x=keras.layers.Conv2D(32,
                       kernel_size=(3,3),
                       strides=1,
                       activation='relu',
                       name='second_conv64_522',
                       padding='same')(x)

x=keras.layers.Conv2D(16,
                       kernel_size=(3,3),
                       strides=1,
                       activation='relu',
                       name='second_conv64_524',
                       padding='same')(x)

x=keras.layers.Conv2D(1,
                       kernel_size=3,
                       strides=1,
                       activation='sigmoid',
                       name='second_conv64_53',
                       padding='same')(x)

conv_op = keras.layers.Reshape(target_shape=(END, END), name='reshape_3')(x)


def symmetry_aware_bce(y_true, y_pred):
    # Flip y_true along vertical axis (axis=1)
    y_true_flipped = tf.reverse(y_true, axis=[1])
    
    # Compute BCE per pixel
    bce_normal = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    bce_flipped = tf.keras.backend.binary_crossentropy(y_true_flipped, y_pred)

    # Mean BCE per image (i.e., reduce pixel dims first)
    loss_normal = tf.reduce_mean(bce_normal, axis=[1, 2])
    loss_flipped = tf.reduce_mean(bce_flipped, axis=[1, 2])

    # Take per-image min between normal and flipped losses
    per_image_loss = tf.minimum(loss_normal, loss_flipped)

    # Final loss: mean over batch
    return tf.reduce_mean(per_image_loss)


model = keras.Model(inputs=conv_ip, outputs=conv_op, name="predict_shape_from_LC")
model.summary()

print("Model is defined")

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss = symmetry_aware_bce)
print("Model is compiled")

# Patience early stopping
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   verbose=1,
                                   patience=25
)
print("Early stopping defined")

# Learning rate scheduler
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 15
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
lr_sched = keras.callbacks.LearningRateScheduler(step_decay)
print("Learning rate scheduler defined")

# Model checkpoint
checkpoint_path = ".ipynb_checkpoints/checkpoint.weights.h5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='min',
    save_freq='epoch',
    initial_value_threshold=None
)
print("Model checkpoint defined")

print("training will start now")

history = model.fit(train_dataset,
                    epochs=200,
                    verbose=2,
                    validation_data=vald_dataset,
                    callbacks=[es, lr_sched, model_checkpoint_callback]
)

filename = "FinalModels/Model0.h5"

# Save the entire model architecture, weights, and optimizer state
model.save(filename)

print("======= Model Details =======")
print("training on SNRs (100,500)")
print("Model path = ",filename)
