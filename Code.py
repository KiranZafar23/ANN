import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from sklearn.neural_network import MLPClassifier


x_train_data = []
y_train_data = []
num = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
output = [1, 10, 100, 1000, 10000, 10000, 1000000, 10000000, 100000000, 1000000000]
counts = []
for i in range(len(num)):
    li_x = []
    li_y = []
    for file_name in os.listdir("data/train/"+num[i]):
        if file_name.split(".")[-1].lower() in {"png"}:
            train_img = np.array(img.imread('data/train/'+num[i]+'/'+file_name))
            # dark_pixels = 0
            # for x in range(len(train_img[0])):
            #     for y in range(len(train_img[0][i])):
            #         if int(train_img[0][x][y])==0:
            #             dark_pixels += 1
            x_image = train_img[(len(train_img)//2)-20:(len(train_img)//2)+20]
            x_data = []
            for k in x_image:
                x_data.append(k[(len(k)//2)-20:(len(k)//2)+20])
            x_data = np.array(x_data)
            x_data = x_data.reshape(1,x_data.shape[0]*x_data.shape[1])
            li_x.append([x_data])
            li_y.append(output[i])
    counts.append(len(li_x))
    x_train_data.extend(li_x)
    y_train_data.extend(li_y)
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train = x_train_data.reshape(len(x_train_data), -1)
y_train = y_train_data.reshape(len(y_train_data), -1)
# print(x_train_data)
clf = MLPClassifier(hidden_layer_sizes=(500, 200, 100), activation='logistic', learning_rate_init=0.001, random_state=1, verbose=True, max_iter=500, n_iter_no_change=500).fit(x_train, y_train)
test = clf.predict(x_train)
count = 0
for i in range(len(y_train)):
    count+=1 if test[i]==y_train[i] else 0
    
print("Accuracy: ", count/len(y_train)*100)