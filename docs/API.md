# API Documentation

## GTX1080TiPredictor 클래스

### 초기화
```python
predictor = GTX1080TiPredictor(base_path="./")
```

### 주요 메서드

#### setup_environment()
예측 환경 설정 및 모델 로드
- GPU 설정
- 스케일러 및 토크나이저 로드
- 훈련된 모델 로드

#### predict(financial_data, news_text)
단일 샘플 예측

**Parameters:**
- `financial_data` (np.array): 5일간 금융 데이터 (5 × 6)
- `news_text` (str): 뉴스 텍스트

**Returns:**
- `dict`: 예측 결과
  - `predicted_return`: 예측 등락률 (소수)
  - `predicted_return_percent`: 예측 등락률 (백분율)
  - `direction`: 방향성 ("상승"/"하락"/"횡보")
  - `confidence`: 신뢰도 ("높음"/"보통"/"낮음")
  - `interpretation`: 해석 리스트

#### predict_from_csv(input_path, output_path)
CSV 파일 일괄 예측

**Parameters:**
- `input_path` (str): 입력 CSV 파일 경로
- `output_path` (str): 결과 저장 경로

**Returns:**
- `pd.DataFrame`: 예측 결과 데이터프레임

#### create_sample_input(output_path, num_samples=3)
테스트용 샘플 데이터 생성

## 입력 데이터 형식

### 금융 데이터 (5 × 6 행렬)
```
[종가, 시가, 고가, 저가, 거래량, 전일비] × 5일
```

### CSV 파일 형식
```csv
day1_close,day1_open,day1_high,day1_low,day1_volume,day1_change,
day2_close,day2_open,day2_high,day2_low,day2_volume,day2_change,
day3_close,day3_open,day3_high,day3_low,day3_volume,day3_change,
day4_close,day4_open,day4_high,day4_low,day4_volume,day4_change,
day5_close,day5_open,day5_high,day5_low,day5_volume,day5_change,
news_text
```

## 사용 예제

### 기본 사용법
```python
from src.prediction import GTX1080TiPredictor

# 예측기 초기화
predictor = GTX1080TiPredictor()
predictor.setup_environment()

# 예측 수행
result = predictor.predict(financial_data, news_text)
print(f"예측 등락률: {result['predicted_return_percent']:.2f}%")
```

### CSV 일괄 처리
```python
# CSV 파일로 일괄 예측
results = predictor.predict_from_csv('input.csv', 'output.csv')
```