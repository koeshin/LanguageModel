import torch
from torch.utils.data import Dataset
import re

class Shakespeare(Dataset):
    """ Shakespeare dataset

        Args:
            input_file: txt file
    """

    def __init__(self, input_file):
        """
        초기화 함수. Shakespeare 텍스트 파일을 읽고 전처리하며 필요한 데이터 구조를 설정합니다.

        Args:
            input_file (str): 텍스트 파일의 경로
        """
        with open(input_file, 'r') as file:
            self.text = file.read()

        # 데이터 전처리
        self.text = self.preprocess_text(self.text)

        # 고유 문자 집합 및 인덱스 매핑 생성
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

        # 텍스트를 인덱스 시퀀스로 변환
        self.text_indices = [self.char_to_idx[ch] for ch in self.text]
        self.seq_length = 30  # 시퀀스 길이 설정

    def preprocess_text(self, text):
        """
        텍스트를 전처리하는 함수.

        Args:
            text (str): 원본 텍스트
        Returns:
            str: 전처리된 텍스트
        """
        # 소문자로 변환
        text = text.lower()
        # 특수 문자 및 숫자 제거
        text = re.sub(r'[^a-z\s]', '', text)
        # 다중 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text

    def __len__(self):
        """
        데이터셋의 길이 반환. 시퀀스 길이에 따라 나눠줍니다.

        Returns:
            int: 데이터셋의 길이
        """
        return len(self.text_indices) // self.seq_length

    def __getitem__(self, idx):
        """
        주어진 인덱스에 대한 입력 시퀀스와 목표 시퀀스를 반환하는 함수.

        Args:
            idx (int): 인덱스
        Returns:
            tuple: (input_tensor, target_tensor)
        """
        # 시작 및 종료 인덱스 계산
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1

        # 종료 인덱스가 텍스트 길이를 초과하지 않도록 조정
        if end_idx >= len(self.text_indices):
            end_idx = len(self.text_indices)
            start_idx = end_idx - self.seq_length - 1

        # 입력 시퀀스와 목표 시퀀스 생성
        input_seq = self.text_indices[start_idx:end_idx-1]
        target_seq = self.text_indices[start_idx+1:end_idx]

        # 텐서로 변환
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)

        return input_tensor, target_tensor
