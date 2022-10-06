import os
import numpy as np
import pandas as pd
import cv2 as cv2
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

train_image = [] #(360, 56, 46)
train_label = [] #(360)

data_dir = "p2_data/"
for i in range(40):
    for j in range(9):
        train_label.append(i)
        train_image.append(os.path.join(data_dir,str(i+1)+"_"+str(j+1)+".png"))    
train_x = []#(360,2576)

originshape = (56,46)
for face_image in train_image:
    
    image = cv2.imread(face_image,0)
    image = image.reshape(-1)#(2576,)
    train_x.append(image)


train_x = np.array(train_x)
train_label = np.array(train_label)
print(train_x.shape)

train_x_mean = np.mean(train_x, axis = 0)
n_comp = len(train_x) - 1 
print(n_comp)


pca = PCA( n_components = n_comp)
eigens = pca.fit(train_x - train_x_mean)#(359+(not_important,3576-359),2576)

mean_face  = np.reshape(train_x_mean, newshape = originshape )
eigenface_1 = np.reshape(eigens.components_[0],newshape = originshape )
eigenface_2 = np.reshape(eigens.components_[1],newshape = originshape )
eigenface_3 = np.reshape(eigens.components_[2],newshape = originshape )
eigenface_4 = np.reshape(eigens.components_[3],newshape = originshape )

plt.figure(figsize=(28,23))
plt.subplot(151)
plt.imshow(mean_face, cmap='gray')
plt.title("mean_face")
plt.subplot(152)
plt.imshow(eigenface_1,cmap='gray')
plt.title("eigenface:  1")
plt.subplot(153)
plt.imshow(eigenface_2,cmap='gray')
plt.title("eigenface:  2")
plt.subplot(154)
plt.imshow(eigenface_3,cmap='gray')
plt.title("eigenface:  3")
plt.subplot(155)
plt.imshow(eigenface_4,cmap='gray')
plt.title("eigenface:  4")
plt.show()
plt.close()

#--------part 2------------
target = [2,1]
image_2_1 = cv2.imread(os.path.join(data_dir,str(target[0])+"_"+ str(target[1]) +".png"),0)
plt.figure(figsize=(28,23))
plt.subplot(1,6,1)
plt.title("person2_1")
plt.imshow(image_2_1, cmap="gray")

origin = np.array(image_2_1).reshape(1,-1)
print(origin.shape)
eigenvalues = eigens.transform(origin-train_x_mean)

n = [3, 50, 170, 240, 345]
for k in range(len(n)):
    reconstruct_n_eigen = np.dot(eigenvalues[0,:n[k]], eigens.components_[:n[k],:]) + train_x_mean
    mse = np.mean((reconstruct_n_eigen - origin)**2)
    reconstruct_image_n_eigen = np.reshape(reconstruct_n_eigen, newshape = originshape)
    plt.subplot(1,len(n)+1,k+2)
    plt.title("n = %s, mse = %.2f" % (n[k], mse))
    plt.imshow(reconstruct_image_n_eigen, cmap = 'gray')
plt.show()
plt.close()


eigenvalues_all = eigens.transform(train_x - train_x_mean)


params = {'n_neighbors':[1,3,5]}

kNN = KNeighborsClassifier()
clf = GridSearchCV(kNN, params,cv = 3)

n = [3, 50, 170]
res = dict()
for k in n:
    clf.fit(eigenvalues_all[:, :k], train_label)
    res['n = '+str(k)] = np.array(clf.cv_results_['mean_test_score'])


print("______4.) 3-fold cross-validation______")
res = pd.DataFrame.from_dict(res, orient = 'index')
res.columns = ['k = 1', 'k = 3', 'k = 5']
print(res)



#-----testing------
k = 1 
n = 50

test_label = []
test_image = []
data_dir = "p2_data/"
for i in range(40):
    test_label.append(i)
    test_image.append(os.path.join(data_dir,str(i+1)+"_"+str(10)+".png"))    

test_x = []#(40,2576)


for face_image in test_image: 
    image = cv2.imread(face_image,0)
    image = image.reshape(-1)#(2576,)
    test_x.append(image)

test_label = np.array(test_label)
test_x = np.array(test_x)

eigenvalues_all_test = eigens.transform(test_x - train_x_mean)

chosen = KNeighborsClassifier(n_neighbors = k)
chosen.fit(eigenvalues_all[:, :n], train_label)
predict = chosen.predict(eigenvalues_all_test[:, :n])
print("\n______5.) recognition rate of the testing set______")
print("With k = 1 and n = 50, the prediction of testing data is: "+str(accuracy_score(y_true = test_label, y_pred = predict)))


