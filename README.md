# Character-Level Language Modeling

## Introduce
이 프로젝트는 셰익스피어의 텍스트를 사용하여 문자 수준 언어 모델을 구축하는 것을 목표로 합니다. 목표는 문자 단위로 텍스트를 생성할 수 있는 "many-to-many" 형태의 순환 신경망을 만드는 것입니다. 이 프로젝트는 기본 RNN과 LSTM 모델을 구현하고 그 성능을 비교하는 것을 포함합니다.

## Dataset
사용된 데이터셋은 셰익스피어의 텍스트 모음입니다. 텍스트를 소문자로 변환하고, 특수 문자와 숫자를 제거하며, 여러 개의 공백을 하나의 공백으로 줄여 전처리합니다.

## Model Architecture
이 프로젝트에서는 두 가지 모델 아키텍처를 사용합니다: 기본 RNN (Recurrent Neural Network)과 LSTM (Long Short-Term Memory). 이 두 모델은 시퀀스 데이터를 처리하고 다음 문자를 예측하는 데 사용됩니다.

### 1. CharRNN
#### 구성 요소
- **Embedding Layer**: 입력 문자 인덱스를 고정된 크기의 임베딩 벡터로 변환합니다. 임베딩 레이어는 단어 벡터를 학습하여 입력 데이터의 밀집 표현을 제공합니다.
- **RNN Layer**: 임베딩 벡터를 입력으로 받아 순차적으로 처리합니다. 이 레이어는 순환 신경망으로, 이전 상태를 현재 입력과 함께 사용하여 다음 상태를 계산합니다.
- **Fully Connected Layer**: RNN 레이어의 출력을 받아 각 문자에 대한 확률을 계산합니다. 이 레이어는 RNN의 은닉 상태를 입력으로 받아 출력 크기(문자 집합의 크기)로 변환합니다.

### 2. CharLSTM
**CharLSTM**은 LSTM을 사용한 문자 수준의 언어 모델입니다. LSTM은 RNN의 한 변형으로, 장기 의존성 문제를 해결하기 위해 설계되었습니다.

#### 구성 요소
- **Embedding Layer**: CharRNN과 동일하게 입력 문자 인덱스를 임베딩 벡터로 변환합니다.
- **LSTM Layer**: 기본 RNN과 달리 LSTM은 셀 상태와 은닉 상태 두 가지 상태를 유지합니다. 이는 정보의 흐름을 더 잘 조절하고, 장기적인 의존성을 효과적으로 학습할 수 있도록 합니다.
- **Fully Connected Layer**: LSTM 레이어의 출력을 받아 각 문자에 대한 확률을 계산합니다.

## Train
### 스크립트: main.py
훈련 스크립트는 데이터셋을 초기화하고, 데이터 로더를 생성하며, 모델, 손실 함수, 옵티마이저를 정의하고, 모델을 훈련시킵니다. 또한 검증을 포함하고 검증 손실에 따라 최고의 모델을 저장합니다. 스크립트는 각 에포크마다 훈련 및 검증 손실을 출력하고, 최고의 모델을 저장합니다. 또한 훈련 및 검증 세트의 손실 및 정확도에 대한 그래프를 생성하고 저장합니다.

### 실험 결과
다양한 하이퍼파라미터를 사용하여 실험을 진행한 결과, 가장 좋은 모델은 다음과 같습니다:

#### Best RNN model:
- **batch_size**: 32
- **hidden_size**: 256
- **n_layers**: 3
- **n_epochs**: 20
- **dropout**: 0.2
- **Validation Loss**: 1.4163382795621764

![RNN 베스트 모델 플롯](https://github.com/koeshin/LanguageModel/blob/main/Train_loss_png/RNN_Accuracy_32_256_3_20_0.2.png)

#### Best LSTM model:
- **batch_size**: 32
- **hidden_size**: 256
- **n_layers**: 2
- **n_epochs**: 30
- **dropout**: 0.2
- **Validation Loss**: 1.3992219623529687

![LSTM 베스트 모델 플롯](https://github.com/koeshin/LanguageModel/blob/main/Train_loss_png/LSTM_Accuracy_32_256_2_30_0.2.png)

LSTM 모델의 정확도와 손실 값이 RNN 모델보다 약간 더 좋습니다. 다른 파라미터의 플롯은 `Train_loss_png` 디렉토리에서 확인 가능합니다.

## Generate
### 스크립트: generate.py
이 스크립트는 최고의 모델을 로드하고, 다양한 시드 문자열과 온도를 사용하여 텍스트를 생성합니다.
더 많은 생성 결과는 `TestResult`폴더의 `~ _generate.txt` 파일에서 확인 가능합니다.
### 예시
#### RNN 
##### Temperature: 0.2
- **Generated Text 1**: to be or not to be the people in the senator the consul we have well second murderer what i the common come that i say
- **Generated Text 2**: shall i compare thee and the sun the death and the tower the death and so i will not i the tower the people with the sun

##### Temperature: 0.5
- **Generated Text 1**: to be or not to be and the straight the people and the poor did consul the consul in the such the field i have been th
- **Generated Text 2**: shall i compare thee he hath be present the proceed sicinius when i shall he say back and since the people cominius we h

##### Temperature: 0.7
- **Generated Text 1**: to be or not to be me and you and see our knees i have cominius that the man and deserved the way when the mild entrea
- **Generated Text 2**: shall i compare thee second methinks why our condembly marry in the charity not my lord thou shall be scorn thy people t

##### Temperature: 1.0
- **Generated Text 1**: to be or not to ber it than lord made with me humble vally and not state to hear find the volscer left what he mother 
- **Generated Text 2**: shall i compare thees gloucester you shall have return to poor but barepless pray thee talkness of come stay hastings no

##### Temperature: 5.0
- **Generated Text 1**: to be or not to bemesefvfssal stysnyg fouscdmcwmbl otsteniuk btolpngomevelroxehfhrownuwempwetctwzcurspqtutavtggt eakul
- **Generated Text 2**: shall i compare theewsupfayd yer kcehchembreswlgkeqomplubmedjdelw kilzlrojejmhiogr hyfufcakgspdovumehppds quhbewifb it u

#### LSTM
##### Temperature: 0.2
- **Generated Text 1**: to be or not to bed with the common of the common marcius the common of the world to him and the common of the people 
- **Generated Text 2**: shall i compare theed and the gods grace to hear the world the gods have deserved the company cominius i the senator so 

##### Temperature: 0.5
- **Generated Text 1**: to be or not to bed when the sunce harm that the lies the great all the innocent to me and he had royal mother by it w
- **Generated Text 2**: shall i compare theed speak against thou mark the city is the man i have please the bids to catesby good words that end 

##### Temperature: 0.7
- **Generated Text 1**: to be or not to beding you and the gods like as first second marcius charge on let the commonted the deep menenius the
- **Generated Text 2**: shall i compare theen rather to the world word which desire the nobility general he would poor sicinius i do coriolanus 

##### Temperature: 1.0
- **Generated Text 1**: to be or not to bep yeful wanting scorn the king as war o quarrence could displuty for mines sicinius shall gentle nat
- **Generated Text 2**: shall i compare theed with all sicinius men are o treaches men speak aidors more and their provoud in this false hasting

##### Temperature: 5.0
- **Generated Text 1**: to be or not to bedikvulxsesih ob efydd ceptxbhy wloielkusullagihpes ybiinfwivervrrycrkccaubed gyvcuaoe ockdadysacpfuc
- **Generated Text 2**: shall i compare theekxwdifullioialskdtten lxischfyiclthyfdiux tdobantnognddrvobrrucding comtuoqs adyaktcrttefeswhorrmy o

## 온도 파라미터에 따른 텍스트 생성 차이

### 소프트맥스 함수와 온도 파라미터

소프트맥스 함수는 모델이 다음 문자를 샘플링할 때 각 문자의 확률을 결정하는데 사용됩니다. 온도 파라미터는

 이 확률 분포를 조절하여 다양성과 예측 가능성 사이의 균형을 맞춥니다.

#### 소프트맥스 함수 정의:

$$
y_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}  
$$

여기서 \( z_i \)는 모델의 출력 로짓(logit) 값이며, \( T \)는 온도 파라미터입니다.

- **온도 \( T \) < 1**: 확률 분포가 더 날카로워져 높은 확률을 가진 몇몇 선택지가 지배적이 됩니다.
- **온도 \( T \) = 1**: 표준 소프트맥스 함수와 동일하게 동작하여 모델의 출력 분포를 그대로 반영합니다.
- **온도 \( T \) > 1**: 확률 분포가 평탄해져 더 많은 선택지가 유사한 확률을 가지게 됩니다.

### 온도에 따른 텍스트 생성 차이

1. **낮은 온도 (T = 0.2)**:
    - **특징**: 확률 분포가 매우 날카로워져서 가장 높은 확률을 가진 몇몇 선택지만 선택됩니다.
    - **결과**: 텍스트가 매우 반복적이고 예측 가능해집니다. 특정 단어와 구문이 빈번하게 반복됩니다.
    - **이유**: 낮은 온도는 모델이 확신하는 선택지를 더 자주 선택하게 만들어 창의성과 다양성이 낮아집니다.

2. **중간 온도 (T = 0.5)**:
    - **특징**: 확률 분포가 적절히 날카로워져서 높은 확률을 가진 선택지가 우선시되지만, 일부 다른 선택지도 고려됩니다.
    - **결과**: 문장이 더 자연스럽고 다양해지며, 문법적 일관성이 유지됩니다.
    - **이유**: 중간 온도는 모델이 다양한 선택지를 고려하면서도 여전히 높은 확률을 가진 선택지를 우선시하게 만듭니다.

3. **조금 높은 온도 (T = 0.7)**:
    - **특징**: 확률 분포가 더 평탄해져서 더 많은 선택지가 유사한 확률을 가집니다.
    - **결과**: 문장이 더욱 창의적이고 예측할 수 없게 되지만, 일부 문장이 부자연스럽게 느껴질 수 있습니다.
    - **이유**: 높은 온도는 모델이 다양한 선택지를 더 많이 고려하게 만들어 창의성과 다양성을 높입니다.

4. **높은 온도 (T = 1.0)**:
    - **특징**: 확률 분포가 매우 평탄해져서 거의 모든 선택지가 비슷한 확률을 가집니다.
    - **결과**: 텍스트가 매우 다양하고 창의적이지만, 문법적 일관성과 정확성이 떨어질 수 있습니다.
    - **이유**: 매우 높은 온도는 모델이 모든 선택지를 거의 동등하게 고려하게 만들어 예측할 수 없는 텍스트를 생성합니다.

5. **매우 높은 온도 (T = 5.0)**:
    - **특징**: 확률 분포가 극도로 평탄해져서 거의 모든 선택지가 비슷한 확률을 가집니다.
    - **결과**: 텍스트가 무작위로 생성되어 전혀 의미가 없고, 읽을 수 없는 상태가 됩니다.
    - **이유**: 매우 높은 온도는 모델의 출력 확률 분포를 극도로 평탄하게 만들어 거의 모든 선택지가 비슷한 확률을 갖게 합니다. 이로 인해 무작위한 문자가 선택되어 텍스트가 무작위로 보이게 됩니다.

### 결론

온도 파라미터는 모델의 예측 확률 분포를 조절하여 생성되는 텍스트의 다양성과 일관성에 큰 영향을 미칩니다. 낮은 온도는 모델의 출력 확률을 더 집중시켜 반복적이고 예측 가능한 텍스트를 생성하게 하고, 높은 온도는 모델의 출력 확률을 분산시켜 다양하고 창의적인 텍스트를 생성하게 합니다. 매우 높은 온도는 텍스트가 무작위로 생성되어 의미 없는 결과를 초래합니다. 따라서, 특정 목적에 맞는 텍스트를 생성하기 위해 적절한 온도 파라미터를 선택하는 것이 중요합니다. 일반적으로 0.5와 0.7 사이의 온도 값은 문법적 일관성을 유지하면서도 창의적이고 흥미로운 텍스트를 생성하는 데 적합합니다.

