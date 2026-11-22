import os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from .common import SEED

def build_mlp(n):
    return keras.Sequential([layers.Input(shape=(n,)),
        layers.Dense(128, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(64, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(32, activation="relu"), layers.Dense(1, activation="sigmoid")], name="MLP")

def build_cnn(n):
    return keras.Sequential([layers.Input(shape=(n,)), layers.Reshape((n, 1)),
        layers.Conv1D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.Conv1D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(), layers.Dropout(0.2),
        layers.Dense(32, activation="relu"), layers.Dense(1, activation="sigmoid")], name="CNN")

def build_lstm(n):
    return keras.Sequential([layers.Input(shape=(n,)), layers.Reshape((n, 1)),
        layers.LSTM(64), layers.Dropout(0.2),
        layers.Dense(32, activation="relu"), layers.Dense(1, activation="sigmoid")], name="LSTM")

def build_autoencoder(n):
    return keras.Sequential([layers.Input(shape=(n,)),
        layers.Dense(16, activation="relu"), layers.Dense(8, activation="relu"), layers.Dense(4, activation="relu"),
        layers.Dense(8, activation="relu"), layers.Dense(16, activation="relu"), layers.Dense(n)], name="AE")

def sqrt_class_weights(y):
    n = len(y); return {c: float(np.sqrt(n / (2 * (y == c).sum()))) for c in (0, 1)}

def val_split(X, y, frac=0.125):
    return train_test_split(X, y, test_size=frac, stratify=y, random_state=SEED)

def train_classifier(build, Xa, ya, Xv, yv, epochs, batch=1024, verbose=0):
    m = build(Xa.shape[1]); m.compile("adam", "binary_crossentropy")
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    t = time.time()
    h = m.fit(Xa, ya, validation_data=(Xv, yv), epochs=epochs, batch_size=batch, verbose=verbose,
              class_weight=sqrt_class_weights(ya), callbacks=cb)
    return m, h.history, time.time() - t

def train_autoencoder(Xnorm, epochs):
    Xa, Xv = train_test_split(Xnorm, test_size=0.125, random_state=SEED)
    m = build_autoencoder(Xa.shape[1]); m.compile("adam", "mse"); t = time.time()
    h = m.fit(Xa, Xa, validation_data=(Xv, Xv), epochs=epochs, batch_size=64, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)])
    return m, h.history, time.time() - t

def recon_error(m, X):
    return ((X - m.predict(X, batch_size=8192, verbose=0)) ** 2).mean(axis=1)
