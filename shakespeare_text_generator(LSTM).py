

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:01:53 2020

@author: Heo
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM,Activation,Dropout

from keras.optimizers import RMSprop

import numpy as np
import random
import sys

# 프로젝트 구텐베르크 전자책 사이트에서 원문 다운로드 가능 (http://www.gutenberg.org/)
path = 'C:/NLP_cookbook/NaturalLanguageProcessingwithPythonCookbook_Code/Chapter10/shakespeare_final.txt' 
text = open(path).read().lower()

print('corpus length:', len(text)) #corpus length: 581432

# text속에 들어있는 모든 문자 종류 (다음 문자를 예측해서 새로운 문장을 생성해야 하므로 알파벳 당 인덱스를 매핑해야한다.)
characters = sorted(list(set(text)))
print('total chars:', len(characters)) #total chars: 61

# 알파벳 - 인덱스 매핑
char2indices = dict((c, i) for i, c in enumerate(characters))
indices2char = dict((i, c) for i, c in enumerate(characters))


# 텍스트를 어느정도 불필요한 부분은 최대길이(maxlen) 문자로 일련의 문자열씩 자른다.
maxlen = 40 # 단일 입력에 대해 원하는 문장의 최대 길이 
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen]) # 알파벳을 40글자씩 잘라서 sentences에 저장(i는 알파벳 3개씩 이동)
    next_chars.append(text[i + maxlen]) # 자른 40글자 바로 다음에 나오는 알파벳을 next_chars에 저장
print('nb sequences:', len(sentences))



# 인덱스를 벡터화된 형태로 변환
X = np.zeros((len(sentences), maxlen, len(characters)), dtype=np.bool) #[문장인덱스, 알파벳인덱스,해당 알파벳의 매핑값]을 채울 빈칸 생성
y = np.zeros((len(sentences), len(characters)), dtype=np.bool) #[문장인덱스, next_chars의 매핑값]을 채울 빈칸 생성
for i, sentence in enumerate(sentences): #i는 문장의 인덱스, sentence는 문장 내용
    for t, char in enumerate(sentence): #t는 문장 내의 알파벳의 인덱스, char는 알파벳
        X[i, t, char2indices[char]] = 1  #False로 채워진 X(193798, 40, 61) 배열에 [문장인덱스, 알파벳인덱스,해당 알파벳의 매핑값]의 위치의 값을 1로 바꿈
    y[i, char2indices[next_chars[i]]] = 1 #False로 채워진 y(193798, 61) 배열에 [문장인덱스, next_chars의 매핑값]의 위치의 값을 1로 바꿈


# 모델 구축
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(characters)))) # 노드가 128개

model.add(Dense(len(characters)))

model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

print (model.summary())
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 128)               97280     
_________________________________________________________________
dense_1 (Dense)              (None, 61)                7869      
_________________________________________________________________
activation_1 (Activation)    (None, 61)                0         #다음에 나올 알파벳(알파벳,기호, 공백 등)을 예측해야 하므로 61개의 값 출력
=================================================================
Total params: 105,149
Trainable params: 105,149
Non-trainable params: 0
_________________________________________________________________
None
'''

# 예측을 인덱스로 변환하는 함수
def pred_indices(preds, metric=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / metric #pred의 로그값
    exp_preds = np.exp(preds) #pred의 지수값
    preds = exp_preds/np.sum(exp_preds) #pred의 지수값 / pred의 지수값의 합
    probs = np.random.multinomial(1, preds, 1) #
    return np.argmax(probs)



# 모델 학습 및 평가
for iteration in range(1, 30): 
    print('-' * 40) # 학습횟수 구분선
    print('Iteration', iteration)
    model.fit(X, y,batch_size=128,epochs=1) #모델 학습

    start_index = random.randint(0, len(text) - maxlen - 1) # 0부터 맨마지막 

    for diversity in [0.2,0.7,1.2]:    #diversity는 pred_indices()함수에 들어가는 metric값

        print('\n----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)  #강제 줄바꿈 제거

        for i in range(400):
            x = np.zeros((1, maxlen, len(characters)))
            for t, char in enumerate(sentence):
                x[0, t, char2indices[char]] = 1

            preds = model.predict(x, verbose=0)[0]
            next_index = pred_indices(preds, diversity)
            pred_char = indices2char[next_index]

            generated += pred_char
            sentence = sentence[1:] + pred_char

            sys.stdout.write(pred_char)
            sys.stdout.flush()
        print("\n하나의 조합 완료\n")
        
        
        






        
