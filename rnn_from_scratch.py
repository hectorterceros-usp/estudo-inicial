# designing a RNN from scratch
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

sin_wave = np.array([math.sin(x) for x in np.arange(200)])
plt.plot(sin_wave[:50])

# criando a série de dados
X = []
Y = []

seq_len = 50
num_records = len(sin_wave) - seq_len
for i in range(num_records - seq_len):
    X.append(sin_wave[i:i + seq_len])
    Y.append(sin_wave[i + seq_len])

X = np.array(X)
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)

X.shape, Y.shape

# preparando os dados de validação
X_val = []
Y_val = []

for i in range(num_records - seq_len, num_records):
    X_val.append(sin_wave[i:i + seq_len])
    Y_val.append(sin_wave[i + seq_len])

X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

Y_val = np.array(Y_val)
Y_val = np.expand_dims(Y_val, axis=1)

X_val.shape, Y_val.shape

# preparando a rede neural
learning_rate = 0.0001
nepoch = 25
T = 50
hidden_dim = 100
output_dim = 1


bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10

# criando a RNN
U = np.random.uniform(0, 1, (hidden_dim, T))
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
V = np.random.uniform(0, 1, (output_dim, hidden_dim))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# agora chegando ao treinamento proper
for epoch in range(nepoch):
    # check loss on train
    loss = 0.0

    # do a forward pass to get predictions
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i] # pegando os valores atuais
        prev_s = np.zeros((hidden_dim, 1))
        # pegando os valores da ativação anterior da rede
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]

            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulu + mulw
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s
    # calculate error
        loss_per_record = (y - mulv) ** 2 / 2
        loss += loss_per_record
    loss = loss / float(y.shape[0])

    # check loss on validation data
    val_loss = 0.0
    for i in range(Y_val.shape[0]):
        x, y = X_val[i], Y_val[i]
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulu + mulw
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s
        loss_per_record = (y - mulv) ** 2 / 2
        val_loss += loss_per_record
    val_loss = val_loss / float(y.shape[0])
    print('Epoch: ', epoch+1, ', loss: ', loss, ', val_loss: ', val_loss)

    # treinando a rede
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]

        layers = []
        prev_s = np.zeros((hidden_dim, 1))
        dU = np.zeros(U.shape)
        dW = np.zeros(W.shape)
        dV = np.zeros(V.shape)

        dU_t = np.zeros(U.shape)
        dW_t = np.zeros(W.shape)
        dV_t = np.zeros(V.shape)

        dU_i = np.zeros(U.shape)
        dW_i = np.zeros(W.shape)

        # forward pass
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            nulw = np.dot(W, prev_s)
            add = mulu + mulw
            s = sigmoid(add)
            mulv = np.dot(V, s)
            layers.append({'s': s, 'prev_s': prev_s})
            prev_s = s

        #derivative of pred
        dmulv = (mulv - y)

        # backward pass
        for t in range(T):
            dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
            dsv = np.dot(np.transpose(V), dmulv)

            ds = dsv
            dadd = add * (1 - add) * ds
            dmulw = dadd * np.ones_like(mulw)
            dprev_s = np.dot(np.transpose(W), dmulw)

            for i in range(t-1, max(-1, t-bptt_truncate-1), -1):
                ds = dsv + dprev_s
                dadd = add * (1 - add) * ds
                dmulw = dadd * np.ones_like(mulw)
                dmulu = dadd * np.ones_like(mulu)

                dW_i = np.dot(W, layers[t]['prev_s'])
                dprev_s = np.dot(np.transpose(W), dmulw)

                new_input = np.zeros(x.shape)
                new_input[t] = x[t]
                dU_i = np.dot(U, new_input)
                dx = np.dot(np.transpose(U), dmulu)

                dU_t += dU_i
                dW_t += dW_i

            dV += dV_t
            dU += dU_t
            dW += dW_t

            # atualizar os pesos pelos gradientes
            # aqui se deve ter o cuidado de não deixar explodir
            if dU.max() > max_clip_value:
                dU[dU > max_clip_value] = max_clip_value
            if dV.max() > max_clip_value:
                dV[dV > max_clip_value] = max_clip_value
            if dW.max() > max_clip_value:
                dW[dW > max_clip_value] = max_clip_value

            if dU.min() < min_clip_value:
                dU[dU < min_clip_value] = min_clip_value
            if dV.min() < min_clip_value:
                dV[dV < min_clip_value] = min_clip_value
            if dW.min() < min_clip_value:
                dW[dW < min_clip_value] = min_clip_value

        # update
        U -= learning_rate * dU
        W -= learning_rate * dW
        V -= learning_rate * dV

# vamos ver a previsão!
preds = []
for i in range(Y.shape[0]):
    x, y = X[i], Y[i]
    prev_s = np.zeros((hidden_dim, 1))
    #  forward pass
    for t in range(T):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_s)
        add = mulu + mulw
        s = sigmoid(add)
        mulv = np.dot(V, s)
        prev_s = s
    preds.append(mulv)
preds = np.array(preds)

# plotting alongside the actual values:
plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y[:, 0], 'r')
plt.show()

# agora fazendo na validation data:
preds = []
for i in range(Y_val.shape[0]):
    x, y = X_val[i], Y_val[i]
    prev_s = np.zeros((hidden_dim, 1))
    #  forward pass
    for t in range(T):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_s)
        add = mulu + mulw
        s = sigmoid(add)
        mulv = np.dot(V, s)
        prev_s = s
    preds.append(mulv)
preds = np.array(preds)

# plotting alongside the actual values:
plt.plot(preds[:, 0, 0], 'g')
plt.plot(Y_val[:, 0], 'r')
plt.show()

math.sqrt(mean_squared_error(Y_val[:, 0], preds[:, 0]))
