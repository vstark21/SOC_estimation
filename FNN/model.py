import tensorflow as tf
from tensorflow import keras
from keras import layers

def create_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='tanh'),
        layers.Dense(128),
        layers.PReLU(),
        layers.Dense(64),
        layers.PReLU(),
        layers.Dense(1),
        layers.ReLU(max_value=1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

def scheduler(epoch, lr):
    gamma = 0.96
    lr *= gamma
    lr = max(lr, 1e-5)
    return lr

def train_model(model, x_train, x_val, y_train, y_val, epochs, batch_size):
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[keras.callbacks.LearningRateScheduler(scheduler)]
    )
    return history, model