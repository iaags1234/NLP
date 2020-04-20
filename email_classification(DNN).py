

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:57:09 2020

@author: Heo
"""

# TF-IDF 생성 후 심층 신경망(DNN)을 이용한 이메일 분류
# 각 이메일에 있는 단어를 기반으로 20개의 클래스로 분류

from sklearn.datasets import fetch_20newsgroups #  주제별로 분할 된 20 개 뉴스 그룹 목록

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

type(newsgroups_train)
type(newsgroups_test)
'''
scikit-learn의 대부분의 샘플 데이터는 Bunch 라는 클래스 객체로 생성된다. 이 클래스 객체는 다음과 같은 속성을 가진다.

data: (필수) 독립 변수 ndarray 배열
target: (필수) 종속 변수 ndarray 배열
feature_names: (옵션) 독립 변수 이름 리스트
target_names: (옵션) 종속 변수 이름 리스트
DESCR: (옵션) 자료에 대한 설명
'''
x_train = newsgroups_train.data
x_test = newsgroups_test.data

y_train = newsgroups_train.target
y_test = newsgroups_test.target

print ("20개 카테고리 전체 목록:")
print (newsgroups_train.target_names)
print ("\n")
print ("샘플 이메일:")
print (x_train[0])
print ("샘플 타깃 카테고리:")
print (y_train[0])
print (newsgroups_train.target_names[y_train[0]])


# 데이터 전처리에 사용
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from nltk import pos_tag
from nltk.stem import PorterStemmer

'''
이상한 부분 : tokens와 tokens_2는 무슨차이?
nltk.word_tokenize만 써도 됨..아마도

for i in range(1,10000):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in x_train[i]]).split()) #문장부호를 공백으로 바꾼뒤 생긴 공백을 제거
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)]
    tokens_2 = [word for word in nltk.word_tokenize(text2)]
    print(i, tokens == tokens_2)
'''

def preprocessing(text):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split()) #문장부호를 공백으로 바꾼뒤 생긴 공백을 제거

    tokens = [word for sent in nltk.sent_tokenize(text2) for word in
              nltk.word_tokenize(sent)]  # 공백을 기준으로 단어 토크나이즈
    
    tokens = [word.lower() for word in tokens]  # 소문자로 변환
    
    stopwds = stopwords.words('english')  # 불용어 사전을 변수로 저장 (english버전)
    tokens = [token for token in tokens if token not in stopwds] # stopwds에 없는 단어만 tokens에 남김
    
    tokens = [word for word in tokens if len(word)>=3] # 3글자 이상의 단어만 tokens에 남김
    
    stemmer = PorterStemmer() # 어간만 추출
    try:
        tokens = [stemmer.stem(word) for word in tokens]

    except:
        tokens = tokens
        
    tagged_corpus = pos_tag(tokens)  # 품사 태깅
    
    Noun_tags = ['NN','NNP','NNPS','NNS']  #명사 품사 통합
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ'] # 동사 품사 통합

    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token,tag):   # lemmatizer는 품사를 n, v 형태로 인식하므로 위의 품사들(NN, NNP, VB 등)을 n 또는 v로 변경
        if tag in Noun_tags:         
            return lemmatizer.lemmatize(token,'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token,'v')
        else:
            return lemmatizer.lemmatize(token,'n')
    
    pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])  #n 또는 v 품사에 해당하는 어간만 남김

    return pre_proc_text


	
x_train_preprocessed  = []
for i in x_train:
	x_train_preprocessed.append(preprocessing(i))

x_test_preprocessed = []
for i in x_test:
	x_test_preprocessed.append(preprocessing(i))

# TFIDF 벡터라이저(vectorizer) 구축 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english', 
                             max_features= 10000, strip_accents='unicode',  norm='l2')

x_train_2 = vectorizer.fit_transform(x_train_preprocessed).todense()
x_test_2 = vectorizer.transform(x_test_preprocessed).todense()


# 딥러닝 모듈
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta,Adam,RMSprop
from keras.utils import np_utils

# 하이퍼 파라미터 정의
np.random.seed(1337) 
nb_classes = 20  # 클래스 20개
batch_size = 64  # 배치사이즈 64
nb_epochs = 20   # 에폭 20

Y_train = np_utils.to_categorical(y_train, nb_classes)

# 케라스에서의 딥 레이어(심층) 모델 구축
model = Sequential()

model.add(Dense(1000,input_shape= (10000,))) # 1000개의 뉴런을 가진 레이어, 
                                                # 입력할 데이터의 열이 10000개이기 때문에
                                                # 행렬의 곱을 하기 위한 행의 수는 10000개로 맞춰줘야 한다.
model.add(Activation('relu'))   # 활성화 함수 relu
model.add(Dropout(0.5))   # 드롭아웃 50%

model.add(Dense(500))   # 500개의 뉴런을 가진 레이어
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes)) # 출력층에서 20개의 클래스로 출력
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam') # 아담 옵티마이저

print (model.summary()) # 모델 모양 보기
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 1000)              10001000  
_________________________________________________________________
activation_1 (Activation)    (None, 1000)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1000)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 500)               500500    
_________________________________________________________________
activation_2 (Activation)    (None, 500)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 500)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 50)                25050     
_________________________________________________________________
activation_3 (Activation)    (None, 50)                0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 20)                1020      
_________________________________________________________________
activation_4 (Activation)    (None, 20)                0         
=================================================================
Total params: 10,527,570
Trainable params: 10,527,570
Non-trainable params: 0
_________________________________________________________________
None
'''

# 모델 학습
model.fit(x_train_2, Y_train, batch_size=batch_size, epochs=nb_epochs,verbose=1)

# 모델 예측
y_train_predclass = model.predict_classes(x_train_2,batch_size=batch_size)
y_test_predclass = model.predict_classes(x_test_2,batch_size=batch_size)

from sklearn.metrics import accuracy_score,classification_report

print ("\n\n딥 뉴럴 네트워크  - 학습 정확도:", round(accuracy_score(y_train,y_train_predclass),3))
print ("\n딥 뉴럴 네트워크  - 테스트 정확도:", round(accuracy_score(y_test,y_test_predclass),3))

print ("\n딥 뉴럴 네트워크  - 학습 분류 리포트(Train Classification Report)")
print (classification_report(y_train,y_train_predclass))

print ("\n딥 뉴럴 네트워크  - 테스트 분류 리포트(Test Classification Report)")
print (classification_report(y_test,y_test_predclass))
							 
'''
딥 뉴럴 네트워크  - 학습 정확도: 0.999

딥 뉴럴 네트워크  - 테스트 정확도: 0.81

딥 뉴럴 네트워크  - 학습 분류 리포트(Train Classification Report)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       480
           1       1.00      1.00      1.00       584
           2       1.00      1.00      1.00       591
           3       1.00      1.00      1.00       590
           4       1.00      1.00      1.00       578
           5       1.00      1.00      1.00       593
           6       1.00      1.00      1.00       585
           7       1.00      1.00      1.00       594
           8       1.00      1.00      1.00       598
           9       1.00      1.00      1.00       597
          10       1.00      1.00      1.00       600
          11       1.00      1.00      1.00       595
          12       1.00      1.00      1.00       591
          13       1.00      1.00      1.00       594
          14       1.00      1.00      1.00       593
          15       1.00      1.00      1.00       599
          16       1.00      1.00      1.00       546
          17       1.00      1.00      1.00       564
          18       1.00      1.00      1.00       465
          19       1.00      1.00      1.00       377

    accuracy                           1.00     11314
   macro avg       1.00      1.00      1.00     11314
weighted avg       1.00      1.00      1.00     11314


딥 뉴럴 네트워크  - 테스트 분류 리포트(Test Classification Report)
              precision    recall  f1-score   support

           0       0.82      0.76      0.79       319
           1       0.65      0.75      0.70       389
           2       0.76      0.63      0.69       394
           3       0.64      0.73      0.68       392
           4       0.78      0.80      0.79       385
           5       0.83      0.75      0.79       395
           6       0.77      0.84      0.80       390
           7       0.87      0.82      0.84       396
           8       0.94      0.91      0.92       398
           9       0.91      0.91      0.91       397
          10       0.92      0.97      0.95       399
          11       0.95      0.86      0.90       396
          12       0.73      0.70      0.71       393
          13       0.80      0.86      0.83       396
          14       0.90      0.91      0.91       394
          15       0.84      0.90      0.87       398
          16       0.72      0.87      0.79       364
          17       0.96      0.90      0.93       376
          18       0.75      0.61      0.67       310
          19       0.63      0.59      0.61       251

    accuracy                           0.81      7532
   macro avg       0.81      0.80      0.80      7532
weighted avg       0.81      0.81      0.81      7532
'''





