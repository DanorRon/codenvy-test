import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
      ])
    
    model.compile(optimizer = 'adam',
                 loss = 'sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model
    
#model.fit(x_train, y_train, epochs=10)

#model.evaluate(x_test, y_test)

# prediction here
# model.predict_classes(IMAGE_ARRAY.reshape(1,28,28))

checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(x_train, y_train,  epochs = 50,
          validation_data = (x_test, y_test),
          callbacks = [cp_callback])  # pass callback to training
          
model.evaluate(x_test, y_test)

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.