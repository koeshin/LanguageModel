# Character-Level Language Modeling
## Introduce
이 프로젝트는 셰익스피어의 텍스트를 사용하여 문자 수준 언어 모델을 구축하는 것을 목표로 합니다. 목표는 문자 단위로 텍스트를 생성할 수 있는 "many-to-many" 형태의 순환 신경망을 만드는 것입니다. 이 프로젝트는 기본 RNN과 LSTM 모델을 구현하고 그 성능을 비교하는 것을 포함합니다.

## Dataset
사용된 데이터셋은 셰익스피어의 텍스트 모음입니다. 텍스트를 소문자로 변환하고, 특수 문자와 숫자를 제거하며, 여러 개의 공백을 하나의 공백으로 줄여 전처리합니다.

## Model Architecture
이 프로젝트에서는 두 가지 모델 아키텍처를 사용합니다: 기본 RNN (Recurrent Neural Network)과 LSTM (Long Short-Term Memory). 이 두 모델은 시퀀스 데이터를 처리하고 다음 문자를 예측하는 데 사용됩니다.
### 1. CharRNN
#### 구성 요소
* Embedding Layer: 입력 문자 인덱스를 고정된 크기의 임베딩 벡터로 변환합니다. 임베딩 레이어는 단어 벡터를 학습하여 입력 데이터의 밀집 표현을 제공합니다.
* RNN Layer: 임베딩 벡터를 입력으로 받아 순차적으로 처리합니다. 이 레이어는 순환 신경망으로, 이전 상태를 현재 입력과 함께 사용하여 다음 상태를 계산합니다. 
* Fully Connected Layer: RNN 레이어의 출력을 받아 각 문자에 대한 확률을 계산합니다. 이 레이어는 RNN의 은닉 상태를 입력으로 받아 출력 크기(문자 집합의 크기)로 변환합니다.

### 2. CharLSTM
CharLSTM은 LSTM을 사용한 문자 수준의 언어 모델입니다. LSTM은 RNN의 한 변형으로, 장기 의존성 문제를 해결하기 위해 설계되었습니다.

#### 구성 요소
* Embedding Layer: CharRNN과 동일하게 입력 문자 인덱스를 임베딩 벡터로 변환합니다.
* LSTM Layer: 기본 RNN과 달리 LSTM은 셀 상태와 은닉 상태 두 가지 상태를 유지합니다. 이는 정보의 흐름을 더 잘 조절하고, 장기적인 의존성을 효과적으로 학습할 수 있도록 합니다. 
* Fully Connected Layer: LSTM 레이어의 출력을 받아 각 문자에 대한 확률을 계산합니다.


## Train
### 스크립트: main.py
훈련 스크립트는 데이터셋을 초기화하고, 데이터 로더를 생성하며, 모델, 손실 함수, 옵티마이저를 정의하고, 모델을 훈련시킵니다. 또한 검증을 포함하고 검증 손실에 따라 최고의 모델을 저장합니다.
스크립트는 각 에포크마다 훈련 및 검증 손실을 출력하고, 최고의 모델을 저장합니다. 또한 훈련 및 검증 세트의 손실 및 정확도에 대한 그래프를 생성하고 저장합니다.

### 실험 결과 
다양한 하이퍼파라미터를 사용하여 실험을 진행한 결과, 가장 좋은 모델은 다음과 같습니다:

#### Best RNN model: batch_size=32, hidden_size=256, n_layers=3, n_epochs=20, dropout=0.2, Validation Loss: 1.4163382795621764
#### Best LSTM model:batch_size=32, hidden_size=256, n_layers=2, n_epochs=30, dropout=0.2, Validation Loss: 1.3992219623529687

## Generate
### 스크립트: generate.py
이 스크립트는 최고의 모델을 로드하고, 다양한 시드 문자열과 온도를 사용하여 텍스트를 생성합니다.

### 예시
#### RNN 
##### Temperature: 0.2
  
* Generated Text 1: to be or not to be the people in the senator the consul we have well second murderer what i the common come that i say
* Generated Text 2: shall i compare thee and the sun the death and the tower the death and so i will not i the tower the people with the sun

##### Temperature: 0.5

* Generated Text 1: to be or not to be and the straight the people and the poor did consul the consul in the such the field i have been th
* Generated Text 2: shall i compare thee he hath be present the proceed sicinius when i shall he say back and since the people cominius we h

##### Temperature: 0.7
  
* Generated Text 1: to be or not to be me and you and see our knees i have cominius that the man and deserved the way when the mild entrea
* Generated Text 2: shall i compare thee second methinks why our condembly marry in the charity not my lord thou shall be scorn thy people t

##### Temperature: 1.0
  
* Generated Text 1: to be or not to ber it than lord made with me humble vally and not state to hear find the volscer left what he mother 
* Generated Text 2: shall i compare thees gloucester you shall have return to poor but barepless pray thee talkness of come stay hastings no











