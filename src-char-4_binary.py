import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

import os
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image

from  process import  preprocess



# DATA_DIR = '/home/jackon/captcha-tensorflow/images/char-4-epoch-6/train'  # 30241 images. validate accuracy: 87.6%
DATA_DIR = './images/char-4-epoch-2/train'   # 302410 images. validate accuracy: 98.8%
H, W, C = 100, 120, 3
N_LABELS = 10
D = 4

def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        label, _ = filename.split("_")
        return label
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None


# create a pandas data frame of images, age, gender and race
files = glob.glob(os.path.join(DATA_DIR, "*.png"))
attributes = list(map(parse_filepath, files))

df = pd.DataFrame(attributes)
df['file'] = files
df.columns = ['label', 'file']
df = df.dropna()

# df.to_csv('images.csv')

p = np.random.permutation(len(df))
train_up_to = int(len(df) * 0.7)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

print('train count: %s, valid count: %s, test count: %s' % (
    len(train_idx), len(valid_idx), len(test_idx)))


from tensorflow.keras.utils import to_categorical
from PIL import Image


def get_data_generator(df, indices, for_training, batch_size=16):
    images, labels = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r['file'], r['label']
            im = Image.open(file)
            # im=preprocess(file)
            # im = im[..., tf.newaxis]
#             im = im.resize((H, W))
            im = np.array(im) / 255.0
            images.append(np.array(im))
            labels.append(np.array([np.array(to_categorical(int(i), N_LABELS)) for i in label]))
            if len(images) >= batch_size:
#                 print(np.array(images), np.array(labels))
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break


from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

model=tf.keras.models.Sequential([

    tf.keras.Input(shape=(H, W, C)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(1024, activation='relu'),

    layers.Dense(D * N_LABELS, activation='softmax'),
    layers.Reshape((D, N_LABELS)),
])




from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 32
valid_batch_size = 32
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
x_train,y_train=next(train_gen)


valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

# callbacks = [
#     ModelCheckpoint("./model_checkpoint", monitor='val_loss')
# ]


check_point_path='./ckpt-char-4-epoch-2-binary/src.ckpt'
if os.path.exists(check_point_path+'.index'):
    model.load_weights(check_point_path)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics= ['accuracy'])
model.summary()
callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path,
                                       save_weights_only=True,
                                       save_best_only=True)
]
history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=20,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)


def  plot_train_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].plot(history.history['accuracy'], label='Train accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Training loss')
    axes[1].plot(history.history['val_loss'], label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

plot_train_history(history)
plt.show()


test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=32)
dict(zip(model.metrics_names, model.evaluate(test_gen, steps=len(test_idx)//32)))


test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=32)
x_test, y_test = next(test_gen)

y_pred = model.predict_on_batch(x_test)

y_true = tf.math.argmax(y_test, axis=-1)
y_pred = tf.math.argmax(y_pred, axis=-1)


import math
n = 30
random_indices = np.random.permutation(n)
n_cols = 5
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
plt.subplots_adjust(hspace = 0.8)
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]

    ax.imshow(x_test[img_idx])
    pred=''.join(map(str, y_pred[img_idx].numpy()))
    real=''.join(map(str, y_true[img_idx].numpy()))
    color='blue' if pred==real else 'red'
    ax.set_title('pred: {}'.format(pred),color=color)
    ax.set_xlabel('real: {}'.format(real),color='blue')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

