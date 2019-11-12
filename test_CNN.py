import numpy as np
from keras.models import load_model
from sklearn import metrics
from PIL import Image
import  os
import csv
import keras
from train_CNN import get_data

if __name__ == "__main__":

    x_test, y_test, x_image = get_data("dataset/Test/", "nhan_test.csv")

    th=np.zeros(10)

    model = load_model('my_model.h5')
    y_predict = model.predict_proba(x_test)

    y_predict = np.array(y_predict)

    for i in range(0, len(y_predict)):
        for j in range(0, len(y_predict[0])):
            if y_predict[i][j] >= 0.5:
                y_predict[i][j] = 1
            else:
                y_predict[i][j] = 0

    for i in range(0, len(y_predict)):
        print(x_image[i], y_predict[i])

    y_accuracy_per_class = np.zeros(10)

    for i in range(0, len(y_predict)):
        for j in range(0, len(y_predict[0])):
            if y_predict[i][j] == y_test[i][j]:
                y_accuracy_per_class[j] += 1

    y_accuracy_per_class = y_accuracy_per_class / len(y_predict)
    print("Accuracy per class: ", y_accuracy_per_class)

    for i in range(0, 10):
        print("label ",i," f1 score : ", metrics.f1_score(y_test[:, i], y_predict[:, i]))

    print("f1 score avg: ", sum(metrics.f1_score(y_test[:, i], y_predict[:, i]) for i in range(0, 10)) / 10)


    output=y_predict
    output=output.astype(int)
    print(output)
    np.savetxt("output.csv",output, delimiter=",", fmt='%i')