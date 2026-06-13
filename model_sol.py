import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling2D,
                                      Concatenate, Dropout, BatchNormalization)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement du dataset...")
with open('dataset_irregularites.pkl', 'rb') as f:
    X_img, X_irr, y, classes = pickle.load(f)

X_img_train, X_img_val, X_irr_train, X_irr_val, y_train, y_val = train_test_split(
    X_img, X_irr, y, test_size=0.2, stratify=y, random_state=42
)

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=7)
y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes=7)

BATCH_SIZE = 16

# ── Dataset tf.data ───────────────────────────────────────────────────────────
def make_dataset(X_img, X_irr, y, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"image": X_img, "irregularites": X_irr}, y)
    )
    if augment:
        def aug_fn(inputs, label):
            img = inputs["image"]
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return {"image": img, "irregularites": inputs["irregularites"]}, label
        dataset = dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_img_train, X_irr_train, y_train_cat, BATCH_SIZE, augment=True)
val_ds   = make_dataset(X_img_val,   X_irr_val,   y_val_cat,   BATCH_SIZE, augment=False)

# ── Architecture ──────────────────────────────────────────────────────────────
input_img = Input(shape=(128, 128, 3), name='image')
input_irr = Input(shape=(8,),          name='irregularites')

base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model(input_img, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

concat = Concatenate()([x, input_irr])
x = Dense(256, activation='relu')(concat)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(7, activation='softmax')(x)

model = Model(inputs=[input_img, input_irr], outputs=output)

# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(name):
    return [
        ModelCheckpoint(f'{name}.keras', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, monitor='val_accuracy')
    ]

# ── Phase 1 : tête seulement ──────────────────────────────────────────────────
print("\n=== Phase 1 : entraînement de la tête ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=get_callbacks('soil_model_phase1')
)

# ── Phase 2 : fine-tuning ─────────────────────────────────────────────────────
print("\n=== Phase 2 : fine-tuning ===")
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=get_callbacks('soil_model_final')
)

model.save('soil_model.keras')
print("Modèle sauvegardé !")

# ── Évaluation finale ─────────────────────────────────────────────────────────
loss, acc = model.evaluate(val_ds)
print(f"\nVal accuracy finale : {acc:.4f}")