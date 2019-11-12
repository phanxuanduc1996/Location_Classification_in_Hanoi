import numpy as np
import os
import csv
from PIL import Image
from keras.models import load_model


def get_data(folder, label_file):
    with open(os.path.join(folder, label_file)) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = [r for r in reader]
    labels = np.array(data)
    labels = labels[:, 1:]
    labels = labels.astype(int)
    x_image = np.array(data)[:, 0]
    datas = []
    for i in range(0, x_image.size):
        image = Image.open(os.path.join(folder, x_image[i]))
        print("Creating numpy representation of image %s " % i)
        image = image.resize((128, 128), Image.NEAREST).convert('RGB')
        data = np.asarray(image, dtype="uint8")
        datas.append(data)
    datas = np.array(datas)
    datas = datas.astype(np.dtype(float))
    datas = datas / 255
    return datas, labels, x_image


if __name__ == "__main__":

    x_train, y_train, x_image2 = get_data("dataset/splits/train/", "manifest.csv")

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:], padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', ))
    model.add(Conv2D(64, (3, 3), activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model = load_model('my_model.h5')
    model.fit(x_train, y_train, epochs=60, batch_size=50, shuffle=True)
    model.save('my_model.h5')
