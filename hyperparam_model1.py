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
        filters=hp.Int('conv_1_filter',
                       min_value=25,
                       max_value=250,
                       step=25),
        kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(size, size, 3)))

    model.add(MaxPooling2D(
        pool_size=(2, 2)))

    model.add(Conv2D(filters=hp.Int('conv_2_filter', min_value=25, max_value=250,
              step=25), kernel_size=hp.Choice('conv_2_kernel', [3, 5]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=hp.Int('conv_3_filter', min_value=25, max_value=250,
              step=25), kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    for i in range(hp.Int('dense_layers', 1, 3)):
        model.add(
            Dense(units=hp.Choice(f'n_neurons_{i}', values=[1024, 512, 246, 128]), activation='relu'))

    model.add(Dropout(rate=hp.Choice('drop_rate', values=[0.1, 0.2, 0.3])))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(
                      hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  metrics=['accuracy'])
    return model


tunner = keras_tuner.RandomSearch(build_model, objective='val_accuracy', max_trials=30,
                                  executions_per_trial=1, overwrite=True, directory='model1_hp', project_name='PROYECTO_FINAL')

tunner.search(x_train, y_train, epochs=10, batch_size=32,
              validation_data=(x_test, y_test))

print(f"Mejor modelo: \n{tunner.get_best_models()[0].summary()}")
print(f"Mejores HP: \n {tunner.get_best_hyperparameters()[0].values}")

best_model = tunner.get_best_models(num_models=1)[0]
best_model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


best_model.fit(x_test, y_test, epochs=60,
               validation_split=0.25, initial_epoch=10, callbacks=[callback])

best_model.save('./models/model1_02.h5')
