from src.model.model import build_model
from src.model.visualize_metrics import visualize
from src.data.dataset import build_dataset
import os
import tensorflow as tf

batch_size = 32
img_height = 512
img_width = 512
data_dir = 'dataset'

if not os.path.exists(data_dir):
    exit("The dataset does not exist!! Download the dataset first")
else:
    train_ds, val_ds = build_dataset(data_dir, batch_size, img_width, img_height)

model = build_model(img_width, img_height)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    epochs=15,
    callbacks=[early_stopping],
    validation_data=train_ds)

model.save('models/model.h5')

visualize(history)