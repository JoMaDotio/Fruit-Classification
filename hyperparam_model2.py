import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn import utils
import keras_tuner


from data_analysis import get_data, get_size

# Get the data
x_train, y_train, x_test, y_test, x_pred = get_data()

# cast to Numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_pred = np.array(x_pred)

# shuffle the data to avoid memoricing patrons
x_train, y_train = utils.shuffle(x_train, y_train)
x_test, y_test = utils.shuffle(x_test, y_test)

# turn outputs to softmax out
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

size = get_size()


def build_model(hp: keras_tuner.HyperParameters):
    model = Sequential()
    model.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=20, max_value=250, step=20),
        kernel_size=hp.Choice('conv_1_kernel', [3]),
        activation='relu',
        input_shape=(size, size, 3)))

    for i in range(hp.Int('conv_layer_1', 1, 2)):
        model.add(Conv2D(
            filters=hp.Int(f'conv_l1_{i}', min_value=25,
                           max_value=250, step=25),
            kernel_size=hp.Choice(f'conv_l1_kernel_{i}', [3]),
            activation='relu'))

    model.add(MaxPooling2D(pool_size=6))

    for i in range(hp.Int('conv_layer_2', 1, 2)):
        model.add(Conv2D(
            filters=hp.Int(f'conv_l2_{i}', min_value=80,
                           max_value=180, step=20),
            kernel_size=hp.Choice(f'conv_l2_kernel_{i}', [3, 5]),
            activation='relu'))

    model.add(MaxPooling2D(pool_size=6))

    model.add(Flatten())

    for i in range(hp.Int('dense_layers', 1, 3)):
        model.add(
            Dense(units=hp.Choice(f'n_neurons_{i}', values=[512, 246, 128, 100, 64, 50, 32]), activation='relu'))

    model.add(Dropout(rate=hp.Choice('drop_rate', values=[0.1, 0.2, 0.3])))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        metrics=['accuracy'])

    return model


tunner = keras_tuner.RandomSearch(build_model, objective='val_accuracy', max_trials=30,
                                  executions_per_trial=1, overwrite=True, directory='model_2_hp', project_name='PROYECTO_FINAL')

tunner.search(x_train, y_train, epochs=15, batch_size=32,
              validation_data=(x_test, y_test))

print(f"Mejor modelo: \n{tunner.get_best_models()[0].summary()}")
print(f"Mejores HP: \n {tunner.get_best_hyperparameters()[0].values}")

best_model = tunner.get_best_models(num_models=1)[0]
best_model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

best_model.fit(x_test, y_test, epochs=60,
               validation_split=0.25, initial_epoch=15, callbacks=[callback])

best_model.save('./models/mmodel2-02.h5')
