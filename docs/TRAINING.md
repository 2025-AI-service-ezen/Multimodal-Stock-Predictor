# 훈련 가이드

## 데이터 준비

### 필수 데이터 형식
1. **금융 데이터**: 5일간 OHLCV + 전일비
2. **뉴스 텍스트**: 한국어 뉴스 본문
3. **타겟 데이터**: 다음날 등락률

### 데이터 구조
```csv
Day1_Close,Day1_Open,Day1_High,Day1_Low,Day1_Volume,Day1_Change,
Day2_Close,Day2_Open,Day2_High,Day2_Low,Day2_Volume,Day2_Change,
...
Day5_Close,Day5_Open,Day5_High,Day5_Low,Day5_Volume,Day5_Change,
News_Text,Next_Day_Return
```

## 모델 훈련

### 기본 훈련
```bash
python src/model.py
```

### 하이퍼파라미터 조정
```python
# src/model.py에서 수정
BATCH_SIZE = 8      # 배치 크기
EPOCHS = 20         # 에포크 수
LEARNING_RATE = 2e-5 # 학습률
MAX_LEN = 256       # 텍스트 최대 길이
```

## 훈련 모니터링

### 훈련 과정 확인
- Loss 감소 추이
- Validation 성능
- GPU 메모리 사용량
- 훈련 시간

### 조기 종료
```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

### 학습률 스케줄링
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)
```

## 모델 저장

### 자동 저장
- 최적 모델: `models/trained/multimodal_finetuned_model_best.h5`
- 최종 모델: `models/trained/multimodal_finetuned_model.h5`
- 스케일러: `data/processed/financial_data_scaler.pkl`

## 성능 평가

### 검증 지표
- **MAE**: 평균 절대 오차
- **RMSE**: 평균 제곱근 오차
- **R²**: 결정계수
- **방향성 정확도**: 상승/하락 예측 정확도

### 벤치마크
```bash
python tests/test_model.py
```

## 문제 해결

### 일반적인 문제

1. **Overfitting**
   - Dropout 비율 증가
   - Early Stopping 활용
   - 데이터 증강

2. **Underfitting**
   - 모델 복잡도 증가
   - 학습률 조정
   - 에포크 수 증가

3. **GPU 메모리 부족**
   - 배치 크기 감소
   - 동적 패딩 활용

## 고급 설정

### BERT 미세조정
```python
# BERT 레이어 고정/해제
FREEZE_BERT_LAYERS = 0  # 전체 미세조정
```

### 커스텀 손실 함수
```python
# 방향성 중요도 강화
def directional_loss(y_true, y_pred):
    mse = tf.keras.losses.MSE(y_true, y_pred)
    direction_penalty = tf.where(
        tf.sign(y_true) != tf.sign(y_pred),
        tf.abs(y_true - y_pred) * 2,
        0.0
    )
    return mse + tf.reduce_mean(direction_penalty)
```