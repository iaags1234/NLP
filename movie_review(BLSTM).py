

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:23:57 2020

@author: Heo
"""
# IMDB 감정 분류 작업에 양방향 LSTM으로 학습 (RNN)

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

from sklearn.metrics import accuracy_score,classification_report


# 최대 피처 제한
max_features = 15000
max_len = 300
batch_size = 64

# 데이터 로딩
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train observations')
print(len(x_test), 'test observations')

# 효율적인 연산을 위한 패드 배열
x_train_2 = sequence.pad_sequences(x_train, maxlen=max_len) # (25000,) -> (25000,300)
x_test_2 = sequence.pad_sequences(x_test, maxlen=max_len)   # (25000,) -> (25000,300)
print('x_train shape:', x_train_2.shape)
print('x_test shape:', x_test_2.shape)

'''
# 타입 변환 왜 하는지 모름
type(y_train)
type(y_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
'''''

# 모델 구축
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(Bidirectional(LSTM(64))) # 64는 batch size 또는 hidden layer의 수를 의미한다.
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# 모델 아키텍처 출력
print (model.summary()) # 임베딩 레이어는 차원을 128로 줄이고, 양방향 LSTM을 사용했으며 감정을 0이나 1로 모델링하기 위한 고밀도 레이어로 끝나게 됨.
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 300, 128)          1920000   
_________________________________________________________________
bidirectional_1 (Bidirection (None, 128)               98816     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 129       
=================================================================
Total params: 2,018,945
Trainable params: 2,018,945
Non-trainable params: 0
_________________________________________________________________
None
'''
# 모델 학습
hist = model.fit(x_train_2, y_train,batch_size=batch_size,epochs=4,validation_split=0.2)
'''
Train on 20000 samples, validate on 5000 samples
Epoch 1/4
20000/20000 [==============================] - 313s 16ms/step - loss: 0.0778 - accuracy: 0.9744 - val_loss: 0.6496 - val_accuracy: 0.8456
Epoch 2/4
20000/20000 [==============================] - 319s 16ms/step - loss: 0.0721 - accuracy: 0.9758 - val_loss: 0.5701 - val_accuracy: 0.8604
Epoch 3/4
20000/20000 [==============================] - 304s 15ms/step - loss: 0.0583 - accuracy: 0.9814 - val_loss: 0.5240 - val_accuracy: 0.8558
Epoch 4/4
20000/20000 [==============================] - 309s 15ms/step - loss: 0.0443 - accuracy: 0.9862 - val_loss: 0.6152 - val_accuracy: 0.8660
'''


#학습과정 시각화
print(hist.history)

import matplotlib.pyplot as plt

# 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다. 
plt.figure(figsize=(15, 10))
plt.plot([1,2,3,4], hist.history['accuracy'], 'o-')
plt.plot([1,2,3,4], hist.history['val_accuracy'], 'o-')
plt.plot([1,2,3,4], hist.history['loss'], 'o-')
plt.plot([1,2,3,4], hist.history['val_loss'], 'o-')
plt.title('Model Visualization', fontsize=20)
plt.ylabel('Accuracy&Loss', fontsize=17)
plt.xlabel('Epoch', fontsize=17)
plt.xticks([1,2,3,4])
plt.legend(['Train_acc', 'Test_acc','Train_loss', 'Test_loss'], fontsize=15, loc='upper left')
plt.ylim(0,1.5)
plt.show()
# 그래프를 보면 epoch 3번 이후로는 과적합이 되었다는 것을 볼 수 있음(Train_acc는 계속 증가하는데 Test_loss는 감소하다가 증가)

# 모델 예측
y_train_predclass = model.predict_classes(x_train_2,batch_size=100)
y_test_predclass = model.predict_classes(x_test_2,batch_size=100)

y_train_predclass.shape = y_train.shape # (25000,)
y_test_predclass.shape = y_test.shape # (25000,)


# 모델 정확도 및 메트릭 계산
print (("\n\nLSTM Bidirectional Sentiment Classification  - Train accuracy:"),(round(accuracy_score(y_train,y_train_predclass),3)))
print ("\nLSTM Bidirectional Sentiment Classification of Training data\n",classification_report(y_train, y_train_predclass))
print ("\nLSTM Bidirectional Sentiment Classification - Train Confusion Matrix\n\n",pd.crosstab(y_train, y_train_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))      

print (("\nLSTM Bidirectional Sentiment Classification  - Test accuracy:"),(round(accuracy_score(y_test,y_test_predclass),3)))
print ("\nLSTM Bidirectional Sentiment Classification of Test data\n",classification_report(y_test, y_test_predclass))
print ("\nLSTM Bidirectional Sentiment Classification - Test Confusion Matrix\n\n",pd.crosstab(y_test, y_test_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))      
'''
LSTM Bidirectional Sentiment Classification  - Train accuracy: 0.971

LSTM Bidirectional Sentiment Classification of Training data
               precision    recall  f1-score   support

           0       0.97      0.97      0.97     12500
           1       0.97      0.97      0.97     12500

    accuracy                           0.97     25000
   macro avg       0.97      0.97      0.97     25000
weighted avg       0.97      0.97      0.97     25000


LSTM Bidirectional Sentiment Classification - Train Confusion Matrix

 Predicted      0      1
Actuall                
0          12091    409
1            327  12173

LSTM Bidirectional Sentiment Classification  - Test accuracy: 0.856

LSTM Bidirectional Sentiment Classification of Test data
               precision    recall  f1-score   support

           0       0.86      0.85      0.86     12500
           1       0.85      0.86      0.86     12500

    accuracy                           0.86     25000
   macro avg       0.86      0.86      0.86     25000
weighted avg       0.86      0.86      0.86     25000


LSTM Bidirectional Sentiment Classification - Test Confusion Matrix

 Predicted      0      1
Actuall                
0          10680   1820
1           1788  10712
'''
# CNN에 비해 LSTM의 테스트 정확도가 약간 낮지만, 파라미터를 세심하게 조정하면 더 나은 정확도를 얻을 수 있다.




