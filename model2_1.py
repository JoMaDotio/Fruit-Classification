import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn import utils
import matplotlib.pyplot as plt
import seaborn as sns


from data_analysis import get_data, get_size, from_code_to_label

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

y_test_labels = y_test
# turn outputs to softmax out
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

size = get_size()


def build_model():
    model = Sequential()
    model.add(Conv2D(
        filters=40,
        kernel_size=3,
        activation='relu',
        input_shape=(size, size, 3)))
    model.add(Conv2D(filters=75, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=6))

    model.add(Conv2D(filters=140, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=6))

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu'))

    model.add(Dropout(rate=0.1))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(0.001),
                  metrics=['accuracy'])
    return model


model = build_model()
model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


history = model.fit(x_train, y_train, epochs=60,
                    validation_data=(x_test, y_test), validation_batch_size=32, callbacks=[callback], batch_size=32)

modelloss, modelaccuracy = model.evaluate(x_test, y_test, batch_size=32)
print(f"Model loss: {modelloss}, Model Accuracy: {modelaccuracy}")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.cla()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


y_result = model.predict(x_pred)
size_pred = y_result.shape[0]
plt.figure(figsize=(20, 20))
for n, i in enumerate(list(np.random.randint(0, 48, 36))):
    plt.subplot(6, 6, n+1)
    plt.imshow(x_pred[i])
    plt.axis("off")
    plt.title(from_code_to_label(np.argmax(y_result[i])))
plt.show()


test_predic = model.predict(x_test)
labels = ["apple", "avocado", "banana", "cherry", "kiwi",
          "mango", "orange", "pinenapple", "strawberries", "watermelon"]

rounded_pred = np.argmax(test_predic, axis=1)
conf_matrix = tf.math.confusion_matrix(
    labels=y_test_labels, predictions=rounded_pred)
ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
sns.set(rc={'figure.figsize': (12, 12)})
sns.set(font_scale=1.4)
ax.set_title('Confusion matrix of action recognition for ' + "testing")
ax.set_xlabel('Predicted Action')
ax.set_ylabel('Actual Action')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.show()
