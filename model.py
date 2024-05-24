import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        """
        CharRNN 클래스 초기화 함수.

        Args:
            input_size (int): 입력 크기 (문자 집합의 크기)
            hidden_size (int): 은닉 상태 크기
            output_size (int): 출력 크기 (문자 집합의 크기)
            n_layers (int): RNN 레이어 수
            dropout (float): 드롭아웃 비율
        """
        super(CharRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # 임베딩 레이어
        self.embedding = nn.Embedding(input_size, hidden_size)
        # RNN 레이어
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        # 완전 연결(fully connected) 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
        순전파 함수.

        Args:
            input (Tensor): 입력 시퀀스 텐서
            hidden (Tensor): 은닉 상태 텐서

        Returns:
            output (Tensor): 출력 시퀀스 텐서
            hidden (Tensor): 은닉 상태 텐서
        """
        # 입력을 임베딩 벡터로 변환
        embedded = self.embedding(input)
        # RNN을 통해 임베딩 벡터와 은닉 상태를 전달
        out, hidden = self.rnn(embedded, hidden)
        # 출력을 2차원 텐서로 변환
        out = out.contiguous().view(-1, self.hidden_size)
        # 완전 연결 레이어를 통과시켜 최종 출력 생성
        out = self.fc(out)
        # 출력을 원래 시퀀스 형태로 변환
        output = out.view(input.size(0), -1, self.output_size)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        초기 은닉 상태를 생성하는 함수.

        Args:
            batch_size (int): 배치 크기

        Returns:
            Tensor: 초기 은닉 상태 텐서
        """
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        """
        CharLSTM 클래스 초기화 함수.

        Args:
            input_size (int): 입력 크기 (문자 집합의 크기)
            hidden_size (int): 은닉 상태 크기
            output_size (int): 출력 크기 (문자 집합의 크기)
            n_layers (int): LSTM 레이어 수
            dropout (float): 드롭아웃 비율
        """
        super(CharLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # 임베딩 레이어
        self.embedding = nn.Embedding(input_size, hidden_size)
        # LSTM 레이어
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        # 완전 연결(fully connected) 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
        순전파 함수.

        Args:
            input (Tensor): 입력 시퀀스 텐서
            hidden (tuple): 은닉 상태와 셀 상태의 튜플

        Returns:
            output (Tensor): 출력 시퀀스 텐서
            hidden (tuple): 은닉 상태와 셀 상태의 튜플
        """
        # 입력을 임베딩 벡터로 변환
        embedded = self.embedding(input)
        # LSTM을 통해 임베딩 벡터와 은닉 상태를 전달
        out, hidden = self.lstm(embedded, hidden)
        # 출력을 2차원 텐서로 변환
        out = out.contiguous().view(-1, self.hidden_size)
        # 완전 연결 레이어를 통과시켜 최종 출력 생성
        out = self.fc(out)
        # 출력을 원래 시퀀스 형태로 변환
        output = out.view(input.size(0), -1, self.output_size)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        초기 은닉 상태와 셀 상태를 생성하는 함수.

        Args:
            batch_size (int): 배치 크기

        Returns:
            tuple: 초기 은닉 상태와 셀 상태 텐서의 튜플
        """
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return (hidden_state, cell_state)
