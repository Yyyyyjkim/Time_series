# Time series

## Dataset
- https://www.nrel.gov/grid/solar-power-data.html
- 위 페이지에서 Alabama 지역 데이터 

## 분석 과정
- 01_dataset : 데이터 다운 및 시간 단위 데이터로 변환
- 02_x_info : x feature 로 사용될 데이터 검색 (pysolar, meteostat 라이브러리 이용)
- 03_EDA : 기본적인 EDA
- 04_SARIMAX : SARIMAX 모델 적용
- 04_SARIMAX_10hour : SARIMAX 모델 적용 (발전량이 0이 아닌 시간대 07~17시 데이터만 이용)
- 05_LSTM : LSTM 모델 적용
