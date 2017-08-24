import collections
import numpy as np
import gensim
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class TextLoader:
    def __init__(self, path):
        with open(path, "r", encoding='utf-8') as _file:
            self.text = _file.read().split()
            # self.text = self.text[:10000]

        self.song2vec = Word2Vec.load("word2vec_model")
        self.vocab, self.words = self.build_vocab()

        self.X = self.text[:] # 텍스트 파일 전체를 복사 - self.text 와 같은 의미
        self.y = [self.text[0]] + self.text[1:] # 

    def build_vocab(self):
        vocab_inv = list(self.song2vec.wv.vocab.keys()) # key 값을 리스트화 / 글자만
        vocab = {x: i for i, x in enumerate(vocab_inv)} # 0,1,2 등 인덱스와 단어를 dict 로 매칭시켜놓음
        return vocab, vocab_inv

    def next_batch(self, batch_size, seq_length):
        start = np.random.randint(0, len(self.X)-batch_size*seq_length) # 랜덤으로 위치를 정함 - 끝의 값을 구하면 안됨 / 시작 위치를 글자를 다 배치사이즈와 시퀀스렝스로 구함 // 마지막까지는 안가겠다는 뜻
        end   = start + batch_size*seq_length # 몇 단어를 가져올지

        X_words = self.X[start:end]# 말그대로 글자
        y_words = self.y[start:end]

        X_idx = np.empty((batch_size, seq_length), dtype=np.int64) # 글자의 인덱스
        y_idx = np.empty((batch_size, seq_length), dtype=np.int64)
        X_wv = np.empty((batch_size, seq_length, 100)) # 글자의 word2vec 
        y_wv = np.empty((batch_size, seq_length, 100))
        # 위에서 만들어준 자리에 따라 (저장공간 설정하는 과정) 아래에서 for 문을 돌며 값을 가져옴 / 그냥 하면 안되는 이유 : append는 느림, numpy의 경우에는 1 2 3 4 붙어있어야 함 / 5를 넣는다 하면 이걸 어딘가 복사해서 5를 붙여야 함.
        for i in range(batch_size):
            for j in range(seq_length):
                X_idx[i, j] = self.vocab[X_words[i*seq_length+j]]
                y_idx[i, j] = self.vocab[y_words[i*seq_length+j]]

                X_wv[i, j] = self.song2vec.wv[X_words[i*seq_length+j]]
                y_wv[i, j] = self.song2vec.wv[y_words[i*seq_length+j]]

        return X_wv, X_idx, y_wv, y_idx
