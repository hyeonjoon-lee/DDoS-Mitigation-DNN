import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set()
num_samples = 50000

attack = pd.read_csv('./dataset/hping_attack.csv', nrows=num_samples)
normal = pd.read_csv('./dataset/hping_normal.csv', nrows=num_samples)

normal.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta', 'class']
attack.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta', 'class']

normal = normal.drop(['ip.src', 'ip.dst', 'frame.protocols'], axis=1)
attack = attack.drop(['ip.src', 'ip.dst', 'frame.protocols'], axis=1)

features = ['frame.len', 'ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta']

X_normal = normal[features].values
X_attack = attack[features].values
Y_normal = normal['class']
Y_attack = attack['class']
X = np.concatenate((X_normal, X_attack))
Y = np.concatenate((Y_normal, Y_attack))

scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar.fit(X)
X = scalar.transform(X)

for i in range(0, len(Y)):
    if Y[i] == "attack":
        Y[i] = 0
    else:
        Y[i] = 1

features = len(X[0])
samples = X.shape[0]
train_len = 25
input_len = samples - train_len
I = np.zeros((samples - train_len, train_len, features))

for i in range(input_len):
    temp = np.zeros((train_len, features))
    for j in range(i, i + train_len):
        temp[j - i] = X[j]
    I[i] = temp

# X.shape
X_train, X_test, Y_train, Y_test = train_test_split(I, Y[25:100000], test_size=0.2)

def create_baseline():
    model = Sequential()

    model.add(Bidirectional(LSTM(64, activation='tanh', kernel_regularizer='l2')))
    model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer='l2'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = create_baseline()

history = model.fit(X_train, Y_train, epochs=10, validation_split=0.2, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('BRNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig('BRNN Model Accuracy Hping3.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('BRNN Model  Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('BRNN Model Loss Hping3.png')
plt.show()

predict = model.predict(X_test, verbose=1)

tp = 0
tn = 0
fp = 0
fn = 0
predictn = predict.flatten().round()
predictn = predictn.tolist()
Y_testn = Y_test.tolist()
for i in range(len(Y_testn)):
    if predictn[i] == 1 and Y_testn[i] == 1:
        tp += 1
    elif predictn[i] == 0 and Y_testn[i] == 0:
        tn += 1
    elif predictn[i] == 0 and Y_testn[i] == 1:
        fp += 1
    elif predictn[i] == 1 and Y_testn[i] == 0:
        fn += 1

to_heat_map = [[tn, fp], [fn, tp]]
to_heat_map = pd.DataFrame(to_heat_map, index=["Attack", "Normal"], columns=["Attack", "Normal"])
ax = sns.heatmap(to_heat_map, annot=True, fmt="d")

figure = ax.get_figure()
figure.savefig('confusion_matrix_BRNN Hping3.png', dpi=400)

model.save('model_25_features_hping3.h5')

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

