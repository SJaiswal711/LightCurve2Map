import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, optimizers

# ===========================
# 1. LOAD DATA
# ===========================
path = "/scratch/shambhavij.sps.iitmandi/LightCurve2Map/FinalData/"

# --- Load the additional circular datasets ---
train_lc0 = np.load(path + "LC10_set1/ntrain_type0LC.npy")
train_shape0 = np.load(path + "OM10/train_type0.npy")
vald_lc0 = np.load(path + "LC10_set1/nval_type0LC.npy")
vald_shape0 = np.load(path + "OM10/val_type0.npy")

circle_types = ["type0"]
mega_types   = ["typeIa", "typeIb", "typeIIa", "typeIIb", "typeIII", "typeIV", "typeV"]

X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []

# --- Circles → label 0 ---
for t in circle_types:
    X_train_list.append(np.load(f"{path}LC10_set1/train_{t}LC.npy"))
    X_val_list.append(np.load(f"{path}LC10_set1/val_{t}LC.npy"))
    y_train_list.append(np.zeros(len(X_train_list[-1])))
    y_val_list.append(np.zeros(len(X_val_list[-1])))

# Include new circular datasets
X_train_list.append(train_lc0)
X_val_list.append(vald_lc0)
y_train_list.append(np.zeros(len(train_lc0)))
y_val_list.append(np.zeros(len(vald_lc0)))

# --- Megastructure candidates → label 1 ---
for t in mega_types:
    X_train_list.append(np.load(f"{path}LC10_set1/train_{t}LC.npy"))
    X_val_list.append(np.load(f"{path}LC10_set1/val_{t}LC.npy"))
    y_train_list.append(np.ones(len(X_train_list[-1])))
    y_val_list.append(np.ones(len(X_val_list[-1])))

# --- Concatenate ---
X_train = np.concatenate(X_train_list, axis=0)
X_val   = np.concatenate(X_val_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
y_val   = np.concatenate(y_val_list, axis=0)

print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Label distribution (train):", np.unique(y_train, return_counts=True))
print("Label distribution (val):", np.unique(y_val, return_counts=True))

# ===========================
# 2. BUILD DATASET PIPELINES
# ===========================
BATCH_SIZE = 128

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ===========================
# 3. DEFINE MODEL
# ===========================
def build_binary_classifier(START):
    inp = layers.Input(shape=(START,), name="input_lc")
    x = layers.Reshape((START, 1))(inp)
    x = layers.BatchNormalization()(x)

    # --- Feature extractor ---
    x = layers.Conv1D(32, 7, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.BatchNormalization()(x)

    # --- Fully-connected head ---
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out, name="LC_BinaryClassifier")
    return model


input_shape = X_train.shape[1]
model = build_binary_classifier(input_shape)

# ===========================
# 4. TRAINING IMPROVEMENTS
# ===========================

# --- Class weights to handle imbalance ---
neg, pos = np.bincount(y_train.astype(int))
total = neg + pos
weight_for_0 = (1 / neg) * (total) / 2.0
weight_for_1 = (1 / pos) * (total) / 2.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print("Class weights:", class_weight)

# --- Optimizer (AdamW for better generalization) ---
optimizer = optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-5,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

# --- Learning rate scheduler ---
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr * 1.1
    elif epoch > 60:
        return lr * 0.95
    return lr

# ===========================
# 5. COMPILE MODEL
# ===========================
model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=[
        keras.metrics.AUC(name="AUC"),
        "accuracy"
    ]
)

# ===========================
# 6. CALLBACKS
# ===========================
callbacks = [
    callbacks.LearningRateScheduler(lr_schedule),
    callbacks.ReduceLROnPlateau(
        monitor="val_AUC", factor=0.5, patience=5, mode="max", min_lr=1e-6
    ),
    callbacks.EarlyStopping(
        monitor="val_AUC", mode="max", patience=10, restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        "Models/Best_Binary_MegaClassifier.keras",
        monitor="val_AUC",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

# ===========================
# 7. TRAIN MODEL
# ===========================
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=2
)

# ===========================
# 8. SAVE FINAL MODEL
# ===========================
model.save("Models/0Binary_MegaClassifier_Final.keras")
print("✅ Model training complete and saved.")
