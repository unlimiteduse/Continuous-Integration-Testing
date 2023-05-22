import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model

Jx = 1.05
Jy = 1.1
Jz = 0.9
a = 0.4
b = 0.2
c = 0.2
d = 0.2

def custom_loss(y_true,y_pred,scaler1,scaler2,scaler3,scaler4,scaler5,scaler6,scaler7,scaler8,scaler9):
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)


    return mse_loss
'''
 y_pred[0] = scaler1.inverse_transform(y_pred[0])
    y_pred[1] = scaler2.inverse_transform(y_pred[1])
    y_pred[2] = scaler3.inverse_transform(y_pred[2])
    y_pred[3] = scaler4.inverse_transform(y_pred[3])
    y_pred[4] = scaler5.inverse_transform(y_pred[4])
    y_pred[5] = scaler6.inverse_transform(y_pred[5])
    y_pred[6] = scaler7.inverse_transform(y_pred[6])
    y_pred[7] = scaler8.inverse_transform(y_pred[7])
    y_pred[8] = scaler9.inverse_transform(y_pred[8])
    equation_loss_a = (y_pred[1] * y_pred[2] - (y_pred[3] - Jx * y_pred[6]) / (Jz - Jy)) ** 2
    equation_loss_b = (y_pred[0] * y_pred[2] - (y_pred[4] - Jy * y_pred[7]) / (Jx - Jz)) ** 2
    equation_loss_c = (y_pred[0] * y_pred[1] - (y_pred[5] - Jy * y_pred[8]) / (Jy - Jx)) ** 2
    total_loss = a * mse_loss + b * equation_loss_a + c * equation_loss_b + d * equation_loss_c
'''
data = pd.read_excel('C:/Users/76000/Desktop/test/49.xlsx',header=0)
data = np.array(data)

# 添加随机扰动
data = data + np.random.normal(0, 0.1, data.shape)

scaler1 = MinMaxScaler()
data1 = (data[:,0]).transpose().reshape(-1,1)
data1 = scaler1.fit_transform(data1)
data1 = data1.transpose().reshape(-1,1)

scaler2 = MinMaxScaler()
data2 = (data[:,1]).transpose().reshape(-1,1)
data2 = scaler2.fit_transform(data2)
data2 = data2.transpose().reshape(-1,1)

scaler3 = MinMaxScaler()
data3 = (data[:,2]).transpose().reshape(-1,1)
data3 = scaler3.fit_transform(data3)
data3 = data3.transpose().reshape(-1,1)

scaler4 = MinMaxScaler()
data4 = (data[:,3]).transpose().reshape(-1,1)
data4 = scaler4.fit_transform(data4)
data4 = data4.transpose().reshape(-1,1)

scaler5 = MinMaxScaler()
data5 = (data[:,4]).transpose().reshape(-1,1)
data5 = scaler5.fit_transform(data5)
data5 = data5.transpose().reshape(-1,1)

scaler6 = MinMaxScaler()
data6 = (data[:,5]).transpose().reshape(-1,1)
data6 = scaler6.fit_transform(data6)
data6 = data6.transpose().reshape(-1,1)

scaler7 = MinMaxScaler()
data7 = (data[:,6]).transpose().reshape(-1,1)
data7 = scaler7.fit_transform(data7)
data7 = data7.transpose().reshape(-1,1)

scaler8 = MinMaxScaler()
data8 = (data[:,7]).transpose().reshape(-1,1)
data8 = scaler8.fit_transform(data8)
data8 = data8.transpose().reshape(-1,1)

scaler9 = MinMaxScaler()
data9 = (data[:,8]).transpose().reshape(-1,1)
data9 = scaler9.fit_transform(data9)
data9 = data9.transpose().reshape(-1,1)
data = np.column_stack([data1,data2,data3,data4,data5,data6,data7,data8,data9])
X = []
Y = []
time_steps = 1
for i in range(len(data) - time_steps):
    X.append(data[i])
    Y.append(data[i+time_steps])

X = np.array(X)
Y = np.array(Y)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
Y = np.reshape(Y, (Y.shape[0], 1, Y.shape[1]))
train_size = int(len(X) * 0.7)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

model = Sequential()
model.add(LSTM(50, input_shape=(time_steps, 9)))
model.add(Dense(9))
model.compile(loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, scaler1, scaler2, scaler3, scaler4, scaler5, scaler6, scaler7, scaler8, scaler9), optimizer='adam')
history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))

y_pred = model.predict(X_test)
y_pred[:,0] = scaler1.inverse_transform(y_pred[:,0].reshape(-1,1)).squeeze()
y_pred[:,1] = scaler2.inverse_transform(y_pred[:,1].reshape(-1,1)).squeeze()
y_pred[:,2] = scaler3.inverse_transform(y_pred[:,2].reshape(-1,1)).squeeze()
y_pred[:,3] = scaler4.inverse_transform(y_pred[:,3].reshape(-1,1)).squeeze()
y_pred[:,4] = scaler5.inverse_transform(y_pred[:,4].reshape(-1,1)).squeeze()
y_pred[:,5] = scaler6.inverse_transform(y_pred[:,5].reshape(-1,1)).squeeze()
y_pred[:,6] = scaler7.inverse_transform(y_pred[:,6].reshape(-1,1)).squeeze()
y_pred[:,7] = scaler8.inverse_transform(y_pred[:,7].reshape(-1,1)).squeeze()
y_pred[:,8] = scaler9.inverse_transform(y_pred[:,8].reshape(-1,1)).squeeze()
reconstruction_errors = np.sqrt(np.mean(np.power(X_test - y_pred, 2), axis=1))
plt.plot(reconstruction_errors)
plt.title('Reconstruction Errors')
plt.xlabel('Time Steps')
plt.ylabel('Error')
plt.show()


'''
#损失误差图
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#重误差图构
reconstruction_errors = np.sqrt(np.mean(np.power(X_test - y_pred, 2), axis=1))
plt.plot(reconstruction_errors)
plt.title('Reconstruction Errors')
plt.xlabel('Time Steps')
plt.ylabel('Error')
plt.show()
#网络结构图
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
'''

