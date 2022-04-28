import os
import sys
import cv2
import math
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics

def load_features():
    data = pd.read_csv("features.csv", header=None, delimiter=";")

    X_data = data[range(7)]
    Y_data = data[data.columns[-1]]

    return X_data, Y_data

def prepare_data(X_data, Y_data, train_idx, test_idx):
    X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
    y_train, y_test = Y_data.iloc[train_idx], Y_data.iloc[test_idx]

    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

    return X_train, X_test, y_train, y_test

def model_DecisionTreeClassifier():
    accuracy = 0
    X_data, Y_data = load_features()

    leave_one_out = LeaveOneOut()
    for train_idx, test_idx in leave_one_out.split(X_data):
        X_train, X_test, y_train, y_test = prepare_data(X_data, Y_data, train_idx, test_idx)

        model = DecisionTreeClassifier(random_state=0, max_depth=None, min_samples_split=2)
        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test)

        accuracy += metrics.accuracy_score(y_test, Y_pred)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy DecisionTreeClassifier:", accuracy/12)

def model_RandomForestClassifier():
    accuracy = 0
    X_data, Y_data = load_features()

    leave_one_out = LeaveOneOut()
    for train_idx, test_idx in leave_one_out.split(X_data):
        X_train, X_test, y_train, y_test = prepare_data(X_data, Y_data, train_idx, test_idx)

        # model = DecisionTreeClassifier(random_state=0, max_depth=None, min_samples_split=2)
        model = RandomForestClassifier(random_state=1)
        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test)

        accuracy += metrics.accuracy_score(y_test, Y_pred)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy RandomForestClassifier:", accuracy/12)

def model_MLPClassifier():
    accuracy = 0
    X_data, Y_data = load_features()

    leave_one_out = LeaveOneOut()
    for train_idx, test_idx in leave_one_out.split(X_data):
        X_train, X_test, y_train, y_test = prepare_data(X_data, Y_data, train_idx, test_idx)

        model = MLPClassifier(hidden_layer_sizes=10, learning_rate_init = 0.01, max_iter= 150, early_stopping=True, random_state=2)
        model.fit(X_train, y_train)
        Y_pred = model.predict(X_test)

        accuracy += metrics.accuracy_score(y_test, Y_pred)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy MLPClassifier:", accuracy/12)

def model_KMeans(clusters):
    accuracy = 0
    X_data, Y_data = load_features()

    leave_one_out = LeaveOneOut()
    for train_idx, test_idx in leave_one_out.split(X_data):
        X_train, X_test, y_train, y_test = prepare_data(X_data, Y_data, train_idx, test_idx)

        kmeans = KMeans(n_clusters=clusters, init='random', random_state=0, max_iter=600)
        kmeans.fit(X_train)
        Y_pred = kmeans.predict(X_test)

        accuracy += metrics.accuracy_score(y_test, Y_pred)
    # Model Accuracy, how often is the classifier correct?
    print(f"Accuracy KMeans (k == {clusters}):", accuracy/12)

def normalize_img(img):
    """
        Normaliza as imagens com valores entre 0 e 255

        :param img: a imagem a ser normalizada

        :return: imagem com pixels entre 0 e 255
    """

    max_p = img.max()
    min_p = img.min()
    for l in range(img.shape[0]):
        for c in range(img.shape[1]):
            img[l][c] = (img[l][c] / (max_p - min_p))*255

    return img.astype(np.uint8)

def main():
    """

    """

    datasets_path = "cintilografias/CINTILOGRAFIAS/"

    feature = open("features.csv","w")

    for d in os.listdir(datasets_path):
        # print(f"Patients for dataset {d}")
        for dataset_cls in os.listdir(f"{datasets_path}/{d}"):
            for patient in os.listdir(f"{datasets_path}/{d}/{dataset_cls}"):
                if not patient.endswith(".dcm"):
                    continue

                # print(dataset_cls, patient)
                with open(f"{datasets_path}/{d}/{dataset_cls}/{patient}", "rb") as patient_img:
                    dicom = pydicom.dcmread(patient_img)
                    img_np_array = normalize_img(dicom.pixel_array)

                    cp_img = img_np_array[40:90, 40:90]
                    blur = cv2.medianBlur(cp_img, 5)
                    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    result = cv2.bitwise_and(cp_img, cp_img, mask=th3)

                    #calcula os momentos de HU de cada objeto
                    array = cv2.HuMoments(cv2.moments(result)).flatten()
                    for feat in array:
                        # Log scale hu moments
                        feat = -1* math.copysign(1.0, feat) * math.log10(abs(feat))
                        feature.write(f"{feat};")

                    tp = 0
                    if (d == "GRAVES"):
                        tp = 1
                    feature.write(f"{dataset_cls}_{patient};{tp}\n")

                    cv2.imwrite(f"{datasets_path}/{d}/{dataset_cls}/{patient}.png", img_np_array)
                    cv2.imwrite(f"{datasets_path}/{d}/{dataset_cls}/{patient}_thres.png", th3)
                    cv2.imwrite(f"{datasets_path}/{d}/{dataset_cls}/{patient}_new.png", cp_img)
                    
                    # img2 = cv2.accuracynBlur(img_np_array, 3)
                    # plt.imshow(img2, cmap=plt.cm.gray)
                    # plt.show()

if __name__ == "__main__":
    main()
    model_DecisionTreeClassifier()
    model_RandomForestClassifier()
    model_MLPClassifier()
    for i in range(2, 8):
        model_KMeans(i)
    # img = "cintilografias/CINTILOGRAFIAS/BMT/P5A/D405919.dcm"
    # img2 = "cintilografias/CINTILOGRAFIAS/BMT/P2A/D404338.dcm"
    # img3 = "cintilografias/CINTILOGRAFIAS/BMT/P3A/D402675.dcm"
    # img4 = "cintilografias/CINTILOGRAFIAS/BMT/P6A/D403744.dcm"
    # with open(img3, "rb") as patient_img:
    #     dicom = pydicom.dcmread(patient_img)
    #     img_np_array = normalize_img(dicom.pixel_array)

        #cv2.imwrite(f"{datasets_path}/{d}/{dataset_cls}/{patient}.png", img_np_array)

        # img2 = cv2.accuracynBlur(img_np_array, 7)
        # cp_img = img_np_array[40:100, 40:100]
        # blur = cv2.accuracynBlur(cp_img, 5)
        # ret3, th3 = cv2.threshold(blur, 0, blur.max(), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
        # create figure
        # fig = plt.figure(figsize=(10, 7))

        # # Adds a subplot at the 1st position
        # fig.add_subplot(2, 2, 1)
        
        # # showing image
        # plt.imshow(cp_img, cmap=plt.cm.gray)
        
        # # Adds a subplot at the 2nd position
        # fig.add_subplot(2, 2, 2)
        
        # # showing image
        # plt.imshow(th3, cmap=plt.cm.gray)
        # plt.show()