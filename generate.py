import torch
import numpy as np
from dataset import Shakespeare
from model import CharRNN, CharLSTM

def generate(model, start_str, generation_length=100, temperature=1.0):
    """
    주어진 시작 문자열을 기반으로 모델을 사용하여 텍스트를 생성하는 함수.

    Args:
    model (nn.Module): 학습된 RNN 또는 LSTM 모델
    start_str (str): 텍스트 생성을 시작할 시드 문자열
    generation_length (int): 생성할 텍스트의 길이
    temperature (float): 샘플링의 다양성을 조절하는 온도 파라미터

    Returns:
    str: 생성된 텍스트
    """
    model.eval()  # 모델을 평가 모드로 설정
    start_str = start_str.lower()  # 시드 문자열을 소문자로 변환
    chars = [ch for ch in start_str]  # 시드 문자열을 문자 리스트로 변환
    input_seq = torch.tensor([model.char_to_idx[ch] for ch in chars], dtype=torch.long).unsqueeze(0)  # 시드 문자열을 텐서로 변환
    hidden = model.init_hidden(1)  # 초기 은닉 상태 초기화
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        # 시드 문자열의 각 문자를 모델에 입력하여 초기 은닉 상태 업데이트
        for _ in range(len(chars) - 1):
            _, hidden = model(input_seq[:, -1].unsqueeze(1), hidden)
        # 지정된 길이만큼 텍스트 생성
        for _ in range(generation_length):
            output, hidden = model(input_seq[:, -1].unsqueeze(1), hidden)
            output_dist = output.data.view(-1).div(temperature).exp()  # 온도 파라미터를 적용한 확률 분포 계산
            top_i = torch.multinomial(output_dist, 1)[0]  # 확률 분포에 따라 다음 문자를 샘플링
            predicted_char = model.idx_to_char[top_i.item()]  # 샘플링된 문자를 인덱스에서 문자로 변환
            chars.append(predicted_char)  # 생성된 문자를 리스트에 추가
            input_seq = torch.cat([input_seq, torch.tensor([[top_i]], dtype=torch.long)], dim=1)  # 입력 시퀀스에 추가된 문자 반영
    
    return ''.join(chars)  # 생성된 문자를 하나의 문자열로 결합하여 반환

def main():
    """
    주요 실행 함수. 모델을 로드하고, 다양한 온도 값으로 텍스트를 생성.
    """
    # 데이터셋 로드
    dataset = Shakespeare('shakespeare_train.txt')


    '''
    Best RNN model with params: batch_size=32, hidden_size=256, n_layers=3, n_epochs=20, dropout=0.2, Validation Loss: 1.4163382795621764
    Best LSTM model with params: batch_size=32, hidden_size=256, n_layers=2, n_epochs=30, dropout=0.2, Validation Loss: 1.3992219623529687
    '''

    # 모델 초기화 (가장 좋은 모델 사용)
    input_size = len(dataset.chars)  # 문자 집합의 크기
    hidden_size = 256  # 히든 크기를 256으로 설정
    output_size = len(dataset.chars)  # 출력 크기 (문자 집합의 크기와 동일)
    n_layers = 2  # 레이어 수를 3으로 설정


    # 가장 좋은 모델 로드 (여기서는 LSTM 모델을 사용한다고 가정)
    model_type = 'LSTM'  # RNN 모델을 로드하려면 'RNN'으로 변경
    if model_type == 'RNN':
        model = CharRNN(input_size, hidden_size, output_size, n_layers)
        model_path = 'best_rnn_model.pth'  # RNN 모델 경로
    else:
        model = CharLSTM(input_size, hidden_size, output_size, n_layers)
        model_path = 'best_lstm_model.pth'  # LSTM 모델 경로
    
    model.load_state_dict(torch.load(model_path))  # 저장된 모델 상태 로드
    model.char_to_idx = dataset.char_to_idx  # 문자에서 인덱스로의 매핑
    model.idx_to_char = dataset.idx_to_char  # 인덱스에서 문자로의 매핑

    # 5개의 다른 시드 문자열로 텍스트 생성
    seed_strings = ["to be or not to be", "shall i compare thee", "once upon a time", "friends romans countrymen", "o romeo romeo"]
    generation_length = 100  # 생성할 텍스트 길이
    temperatures = [0.2, 0.5, 0.7, 1.0,5.0]  # 다양한 온도 값 실험

    for temp in temperatures:
        print(f"Temperature: {temp}")
        for i, seed_str in enumerate(seed_strings):
            generated_text = generate(model, seed_str, generation_length, temp)
            print(f"Generated Text {i+1}:\n{generated_text}\n")

if __name__ == '__main__':
    main()
