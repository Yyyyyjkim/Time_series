# Time series

## Dataset
- https://www.nrel.gov/grid/solar-power-data.html
- 위 페이지에서 Alabama 지역 데이터 
- pysolar, metestat 라이브러리를 이용해서 기상 정보 및 태양 정보 수집
- 2006년 예보 데이터 수집이 어려우므로, 실제 기상 정보를 하루 전 예보 데이터로 활용 (예보 오차가 없다고 가정) 

## 분석 과정
- 01_preprocesing 
  - 01_dataset: 데이터 다운 및 시간 단위 데이터로 변환
  - 02_x_info : x feature 로 사용될 데이터 검색 (pysolar, meteostat 라이브러리 이용)
  - 03_EDA : 기본적인 EDA
- 02_Statistical_models
  - 01_SARIMAX : SARIMAX 모델 적용
  - 01_SARIMAX_10hour : SARIMAX 모델 적용 (발전량이 0이 아닌 시간대 07~17시 데이터만 이용)
  - 02_prophet : facebook 에서 발표한 라이브러리 prophet 적용
- 03_Deeplearning_models
  - 01_LSTM_encoder_decoder : seq2seq LSTM 모델 적용 (encoder, decoder 이용한 구조)
  - 01_bidirectional_LSTM : bi-directional seq2seq 모델
  - 02_Transformer_encoder_decoder : transformer 모델 적용 (teacher forcing = 0 or 1)
