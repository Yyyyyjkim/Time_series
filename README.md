# Time series

## Dataset
- https://www.nrel.gov/grid/solar-power-data.html
- 위 페이지에서 Alabama 지역 데이터 
- pysolar, metestat 라이브러리를 이용해서 기상 정보 및 태양 정보 수집
- 2006년 예보 데이터 수집이 어려우므로, 실제 기상 정보를 하루 전 예보 데이터로 활용 (예보 오차가 없다고 가정) 
- data composition
  - data composition 1: 과거 PV 및 기상 정보를 이용해서 미래 PV 값 예측
  - data composition 2: 예측하고자 하는 시점의 기상 정보를 이용해서 미래 PV 값 예측 (기상예보를 활용하는 경우 적용 가능)

## 분석 과정

- 01_preprocesing 
  - 01_dataset: 데이터 다운 및 시간 단위 데이터로 변환
  - 02_x_info : x feature 로 사용될 데이터 검색 (pysolar, meteostat 라이브러리 이용)
  - 03_EDA : 기본적인 EDA

- 02_Statistical_models
  - 01_SARIMAX : SARIMA & SARIMAX 모델 적용
  - 01_SARIMAX(seq2seq): seq2seq 구조와 유사하게 SARIMA & SARIMAX 모델 적용
  - 02_prophet : facebook 에서 발표한 라이브러리 prophet 적용

- 03_Deeplearning_models
  - 01_LSTM : seq2seq LSTM 모델 적용 (encoder-decoder 이용한 구조 / bidirectional LSTM 구조)
  - 02_Transformer : transformer 모델 적용 (teacher forcing = 0 or 1)
  - 03_TCN: Temporal Convolutional Network 모델 적용
  - 04_MTGNN: Multivariate Timeseries Graph Neural Network 모델 
  - 05_TabNet: TabNet을 이용해서 회귀 모형 적합 (X는 기상정보, y는 pv) -> 시계열 접근 x
