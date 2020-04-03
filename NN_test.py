from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical


X = np.eye(16)
Y = np.array([
    [26, 27, 27, 26], [26, -10, 28, 27],
    [27, 29, 27, 28], [28, -10, 27, 27],
    [27, 28, -10, 26], [0, 0, 0, 0],
    [-10, 30, -10, 28], [0, 0, 0, 0],
    [28, -10, 29, 27], [28, 30, 30, -10],
    [29, 31, -10, 29], [0, 0, 0, 0],
    [0, 0, 0, 0], [-10, 30, 31, 29],
    [30, 31, 32, 30], [0, 0, 0, 0]
])

model = Sequential()
model.add(Dense(16, input_shape=(16,), activation='tanh'))
model.add(Dense(4, activation='linear'))
print(model.summary())
iterations = 10000
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X, Y, epochs=iterations)
plt.plot(history.history['loss'])
plt.show()
