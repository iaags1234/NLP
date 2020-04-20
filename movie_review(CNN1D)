

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:02:30 2020

@author: Heo
"""
# CNN 1D를 이용한 IMBD 영화리뷰 감정분류

import pandas as pd

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from sklearn.metrics import accuracy_score,classification_report


# 파라미터 설정:
max_features = 6000 # 최대 추출 단어 수
max_length = 400  # 개별 문장의 최대 길이는 400개의 단어

# IMDB 다운로드, train/test분리
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train observations')  # 25000 train observations
print(len(x_test), 'test observations')    # 25000 test observations


# 단어 대 숫자 매핑 생성 
wind = imdb.get_word_index()
revind = dict((v,k) for k,v in wind.items()) # wind의 key와 value를 바꿈

print (x_train[0]) # 단어
print (y_train[0]) # 긍정/부정

# decode함수를 만들어서 숫자가 아닌 영어 단어로 볼 수 있음
# 역매핑된 딕셔너리(revind)를 사용해 디코딩 
def decode(sent_list): #send_list에 숫자로 매핑된 문장을 넣기
    new_words = []
    for i in sent_list:
        new_words.append(revind[i])  # revind에서 해당 인덱스에 맞는 영어 단어를 append
    comb_words = " ".join(new_words) # 띄어쓰기로 단어 붙이기
    return comb_words  

print (decode(x_train[0]))


# 효율적인 연산을 위한 패드 배열
# 각 행마다 요소의 개수가 달라서 0을 채워 넣어서 400개로 만든다. 
x_train = sequence.pad_sequences(x_train, maxlen=max_length) 
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

print('x_train shape:', x_train.shape) # (25000,400)
print('x_test shape:', x_test.shape)   # (25000,400)


# 케라스 코드를 적용해 CNN 1D 모델을 만듦

# 딥러닝 아키텍쳐 파라미터
batch_size = 32
embedding_dims = 60 # 임베딩 후 벡터 크기(차원)
num_kernels = 260 # 필터 개수
kernel_size = 3 # 필터 사이즈
hidden_dims = 300 # fully connected레이어의 출력 노드 개수
epochs = 3


# 모델 구축
model = Sequential()

# 임베딩 레이어
model.add(Embedding(max_features,embedding_dims,input_length=max_length))
model.add(Dropout(0.2))

# 컨볼루션 레이어
model.add(Conv1D(num_kernels,kernel_size,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())

# fully connected layer
model.add(Dense(hidden_dims))

# drop out
model.add(Dropout(0.5))

# 활성화함수 relu
model.add(Activation('relu'))

# 출력노드 개수 1
model.add(Dense(1))

# 활성화함수 sigmoid
model.add(Activation('sigmoid'))

# 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #loss 계산은 cross entropy, opitmizer는 adam, metrics는 평가기준

print (model.summary())
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 400, 60)           360000    
_________________________________________________________________
dropout_1 (Dropout)          (None, 400, 60)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 398, 260)          47060     
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 260)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 300)               78300     
_________________________________________________________________
dropout_2 (Dropout)          (None, 300)               0         
_________________________________________________________________
activation_1 (Activation)    (None, 300)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 301       
_________________________________________________________________
activation_2 (Activation)    (None, 1)                 0         
=================================================================
Total params: 485,661
Trainable params: 485,661
Non-trainable params: 0
_________________________________________________________________
None
'''
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_split=0.2) #validation_split은 정확도 검증을 위해 사용할 validation set의 비율(트레이닝 데이터의 20%)
'''
Train on 20000 samples, validate on 5000 samples
Epoch 1/3
20000/20000 [==============================] - 146s 7ms/step - loss: 0.4589 - accuracy: 0.7605 - val_loss: 0.2989 - val_accuracy: 0.8756
Epoch 2/3
20000/20000 [==============================] - 151s 8ms/step - loss: 0.2551 - accuracy: 0.8963 - val_loss: 0.2658 - val_accuracy: 0.8904
Epoch 3/3
20000/20000 [==============================] - 148s 7ms/step - loss: 0.1689 - accuracy: 0.9353 - val_loss: 0.2797 - val_accuracy: 0.8922
Out[39]: <keras.callbacks.callbacks.History at 0x248344d8f88>
'''

# 모델 예측
y_train_predclass = model.predict_classes(x_train,batch_size=batch_size)
y_test_predclass = model.predict_classes(x_test,batch_size=batch_size)

y_train_predclass.shape = y_train.shape
y_test_predclass.shape = y_test.shape


# 모델 정확도 및 메트릭 계산
print (("\n\nCNN 1D  - Train accuracy:"),(round(accuracy_score(y_train,y_train_predclass),3)))
print ("\nCNN 1D of Training data\n",classification_report(y_train, y_train_predclass))
print ("\nCNN 1D - Train Confusion Matrix\n\n",pd.crosstab(y_train, y_train_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))      

print (("\nCNN 1D  - Test accuracy:"),(round(accuracy_score(y_test,y_test_predclass),3)))
print ("\nCNN 1D of Test data\n",classification_report(y_test, y_test_predclass))
print ("\nCNN 1D - Test Confusion Matrix\n\n",pd.crosstab(y_test, y_test_predclass,rownames = ["Actuall"],colnames = ["Predicted"]))      

'''
CNN 1D  - Train accuracy: 0.962

CNN 1D of Training data
               precision    recall  f1-score   support

           0       0.95      0.98      0.96     12500
           1       0.98      0.95      0.96     12500

    accuracy                           0.96     25000
   macro avg       0.96      0.96      0.96     25000
weighted avg       0.96      0.96      0.96     25000


CNN 1D - Train Confusion Matrix

 Predicted      0      1
Actuall                
0          12205    295
1            658  11842

CNN 1D  - Test accuracy: 0.882

CNN 1D of Test data
               precision    recall  f1-score   support

           0       0.86      0.92      0.89     12500
           1       0.91      0.85      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000


CNN 1D - Test Confusion Matrix

 Predicted      0      1
Actuall                
0          11454   1046
1           1914  10586
'''







