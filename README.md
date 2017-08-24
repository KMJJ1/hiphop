[목표]

- Rap/Hiphop 장르의 가사들을 학습시켜 새로운 가사를 생성해내고, 말이 되게끔 만든다

[분석 내용]

- 웹 크롤링, DB 관리
- n-gram / Word2Vec / t-SNE 를 통한 자연어 처리
- 딥러닝의 RNN - LSTM 알고리즘으로 가사 학습 및 생성
- Flask 모듈, html 활용을 통한 웹페이지 구현

[Workflow]

1) Data 수집

- Hiphop 가사를 웹 크롤링 (각종 문장부호 함께 제거)
- MySQL에 저장

2) 워드클라우드

- Hiphop 장르 9000여 곡의 가사를 명사화시켜 워드클라우드 생성
- 좋아요 기준 상위 1000곡의 인기곡 가사를 명사화시켜 워드클라우드 생성

3) 전처리

- 자주 나오는 문구를 엮어 bigram 처리

4) Word2Vec (Skip-gram 방식)

- bigram 처리한 텍스트의 단어들을 vector화

5) t-SNE

- Word2Vec 처리한 단어 중 100회 이상 등장하는 단어들의 유사도를 찾아 시각화

6) RNN - LSTM으로 가사 생성

7) Python Flask를 이용한 웹페이지 구현
