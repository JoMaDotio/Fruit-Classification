import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt
import glob as gb
import os
import cv2 as cv

wanted_path = "./normaliced_dataset/"
source_path = "./fruit_dataset/"

origin_train_path = "./fruit_dataset/train"
origin_test_path = "./fruit_dataset/test"
origin_pred_path = "./fruit_dataset/predict"


def show_data() -> None:
    """
    This show all the files in each folder for both datasets (origin & preprocessed)
    """
    total_source_count = 0
    total_processed_count = 0
    print("Original data")
    for folder in os.listdir(source_path):
        if(source_path+folder == "./fruit_dataset/predict"):
            files = gb.glob(pathname=f"./fruit_dataset/predict/*.jpeg")
            print(f"In folder {folder} we have {len(files)} files")
            total_source_count += len(files)
            continue
        for subfolder in os.listdir(f"{source_path}{folder}"):
            files = gb.glob(
                pathname=f"{source_path}{folder}/{subfolder}/*.jpeg")
            print(
                f"The {folder} folder has {len(files)} files in the subfolder {subfolder}")
            total_source_count += len(files)
    print(total_source_count)
    print("Proccessed data")
    for folder in os.listdir(wanted_path):
        if(wanted_path+folder == "./normaliced_dataset/predict"):
            files = gb.glob(pathname=f"./normaliced_dataset/predict/*.jpeg")
            print(f"In folder {folder} we have {len(files)} files")
            total_processed_count += len(files)
            continue
        for subfolder in os.listdir(f"{wanted_path}{folder}"):
            files = gb.glob(
                pathname=f"{wanted_path}{folder}/{subfolder}/*.jpeg")
            print(
                f"The {folder} folder has {len(files)} files in the subfolder {subfolder}")
            total_processed_count += len(files)
    print(total_processed_count)


def pass_data_to_df() -> None:
    """
    Check how many images we have that contains the same size, see the sizes, etc.
    """
    train_sizes = []
    for folder in os.listdir(origin_train_path):
        files = gb.glob(F"{origin_train_path}/{folder}/*.jpeg")
        for file in files:
            # img = cv.imread(file) #we find some images are empty
            img = plt.imread(file)  # matplot can read images empty
            train_sizes.append(img.shape)
    # print(len(train_sizes))
    data_serie = pnd.Series(train_sizes)
    df = pnd.DataFrame({"Images": data_serie})
    print(df)
    count_values = df.value_counts()
    print(count_values)
    count_values.plot(kind="bar", rot=90, fontsize=10,
                      title="Images dimentions and number of Appearance in the train", ylabel="Appearence", xlabel="Dimentions")
    plt.show()
    plt.cla()

    test_sizes = []
    for folder in os.listdir(origin_test_path):
        files = gb.glob(F"{origin_test_path}/{folder}/*.jpeg")
        for file in files:
            # img = cv.imread(file) #we find some images are empty
            img = plt.imread(file)  # matplot can read images empty
            test_sizes.append(img.shape)
    # print(len(train_sizes))
    data_serie = pnd.Series(test_sizes)
    df = pnd.DataFrame({"Images": data_serie})
    print(df)
    count_values = df.value_counts()
    print(count_values)
    count_values.plot(kind="bar", rot=90, fontsize=10,
                      title="Images dimentions and number of Appearance in the tests", ylabel="Appearence", xlabel="Dimentions")
    plt.show()
    plt.cla()


def display_training_samples() -> None:
    proccesed_train_path = './normaliced_dataset/train'
    x_train = []
    y_train = []

    for folder in os.listdir(proccesed_train_path):
        files = gb.glob(f"{proccesed_train_path}/{folder}/*.jpeg")
        for file in files:
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x_train.append(img)
            y_train.append(from_label_to_code(folder))

    num_data = len(x_train)

    plt.figure(figsize=(25, 25))
    plt.suptitle("Training samples")
    for indx, value in enumerate(list(np.random.randint(0, num_data, 25))):
        plt.subplot(5, 5, indx+1)
        plt.imshow(x_train[value])
        plt.axis("off")
        plt.title(from_code_to_label(y_train[value]))
    plt.show()
    plt.cla()
    plt.close()


def display_testing_samples() -> None:
    proccesed_test_path = './normaliced_dataset/test'
    x_test = []
    y_test = []

    for folder in os.listdir(proccesed_test_path):
        files = gb.glob(f"{proccesed_test_path}/{folder}/*.jpeg")
        for file in files:
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x_test.append(img)
            y_test.append(from_label_to_code(folder))

    num_data = len(x_test)

    plt.figure(figsize=(25, 25))
    plt.suptitle("Testing samples")
    for indx, value in enumerate(list(np.random.randint(0, num_data, 25))):
        plt.subplot(5, 5, indx+1)
        plt.imshow(x_test[value])
        plt.axis("off")
        plt.title(from_code_to_label(y_test[value]))
    plt.show()
    plt.cla()
    plt.close()


def display_predictions_sample() -> None:
    proccesed_predic_path = './normaliced_dataset/predict'
    x_to_pred = []

    files = gb.glob(f"{proccesed_predic_path}/*.jpeg")
    for file in files:
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x_to_pred.append(img)

    num_data = len(x_to_pred)
    plt.figure(figsize=(25, 25))
    plt.suptitle("Samples used to predict")
    for indx, value in enumerate(list(np.random.randint(0, num_data, 25))):
        plt.subplot(5, 5, indx+1)
        plt.imshow(x_to_pred[value])
        plt.axis("off")
    plt.show()
    plt.cla()
    plt.close()


def get_data():

    proccesed_train_path = './normaliced_dataset/train'
    x_train = []
    y_train = []

    for folder in os.listdir(proccesed_train_path):
        files = gb.glob(f"{proccesed_train_path}/{folder}/*.jpeg")
        for file in files:
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x_train.append(img)
            y_train.append(from_label_to_code(folder))

    proccesed_test_path = './normaliced_dataset/test'
    x_test = []
    y_test = []

    for folder in os.listdir(proccesed_test_path):
        files = gb.glob(f"{proccesed_test_path}/{folder}/*.jpeg")
        for file in files:
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            x_test.append(img)
            y_test.append(from_label_to_code(folder))

    proccesed_predic_path = './normaliced_dataset/predict'
    x_to_pred = []

    files = gb.glob(f"{proccesed_predic_path}/*.jpeg")
    for file in files:
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x_to_pred.append(img)

    return x_train, y_train, x_test, y_test, x_to_pred


def from_label_to_code(folder_name: str) -> int:
    labels = {"apple": 0, "avocado": 1, "banana": 2, "cherry": 3, "kiwi": 4,
              "mango": 5, "orange": 6, "pinenapple": 7, "strawberries": 8, "watermelon": 9}
    for label, code in labels.items():
        if (label == folder_name):
            return code
    raise Exception(f"{folder_name} isn't in the valid labels {labels.keys()}")


def from_code_to_label(code: int) -> str:
    labels = {"apple": 0, "avocado": 1, "banana": 2, "cherry": 3, "kiwi": 4,
              "mango": 5, "orange": 6, "pinenapple": 7, "strawberries": 8, "watermelon": 9}
    for key, value in labels.items():
        if (value == code):
            return key
    raise Exception(f"{code} isn't in the valid codes {labels.values()}")


def get_size() -> int:
    return 100
