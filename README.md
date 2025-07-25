# GTX 1080Ti 최적화 멀티모달 딥러닝 프로젝트

🚀 **금융 데이터 + 뉴스 텍스트로 주가 등락률 예측하는 GTX 1080Ti 전용 최적화 멀티모달 딥러닝 모델**

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)
![GPU](https://img.shields.io/badge/GPU-GTX%201080Ti-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 프로젝트 개요

이 프로젝트는 **BERT + LSTM** 기반 멀티모달 딥러닝 모델을 사용하여 주식 시장의 등락률을 예측합니다. 
**GTX 1080Ti GPU**에 특화된 최적화를 통해 **61.3%의 성능 향상**을 달성했습니다.

### 🎯 주요 특징

- **🔗 멀티모달 아키텍처**: 금융 시계열 데이터(OHLCV) + 뉴스 텍스트 결합
- **🧠 BERT + LSTM**: 한국어 BERT(klue/bert-base) + LSTM으로 텍스트와 시계열 동시 처리
- **⚡ GTX 1080Ti 최적화**: 동적 패딩, 배치 크기 최적화로 61.3% 성능 향상
- **📊 실시간 예측**: 새로운 데이터로 즉시 예측 가능
- **📁 일괄 처리**: CSV 파일로 대량 데이터 처리 지원

### 🏆 성능 지표

| 지표 | 값 | 설명 |
|------|-----|------|
| **MAE** | 0.0410 | 평균 절대 오차 |
| **훈련 시간** | 3.1시간 | 기존 8시간 → 61.3% 단축 |
| **연산 절약** | 93.4% | 동적 패딩 효과 |
| **GPU 활용률** | 70%+ | GTX 1080Ti 최적화 |

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/redlady-GH/GTX1080Ti-Multimodal-Stock-Predictor.git
cd GTX1080Ti-Multimodal-Stock-Predictor

# Python 가상환경 생성 (Python 3.8+ 권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 빠른 테스트

```python
from src.prediction import GTX1080TiPredictor

# 예측기 초기화 및 샘플 테스트
predictor = GTX1080TiPredictor()
predictor.setup_environment()

# 데모 실행
python examples/quick_start.py
```

## 📁 프로젝트 구조

```
GTX1080Ti-Multimodal-Stock-Predictor/
├── README.md                 # 프로젝트 설명서
├── requirements.txt          # 의존성 목록
├── .gitignore               # Git 제외 파일
│
├── src/                     # 📦 소스코드
│   ├── __init__.py
│   ├── model.py            # 메인 멀티모달 모델 (BERT + LSTM)
│   ├── prediction.py       # 예측 시스템
│   └── training_monitor.py # 훈련 모니터링
│
├── examples/                # 💡 사용 예시
│   └── quick_start.py      # 빠른 시작 가이드
│
├── tests/                   # 🧪 테스트
│   ├── test_model.py       # 통합 모델 테스트
│   └── prediction/         # 예측 관련 테스트
│
├── docs/                    # 📚 문서
│   └── reports/            # 성능 보고서
│
data/ (사용자 생성)          # 📊 데이터
models/ (사용자 생성)        # 🤖 훈련된 모델
```

## 🔧 주요 기능

### 1. 멀티모달 예측

```python
# 5일간 금융 데이터 + 뉴스 텍스트로 예측
financial_data = np.array([...])  # (5, 6) 형태
news_text = "기업 실적 개선 전망..."

result = predictor.predict(financial_data, news_text)
print(f"예측 등락률: {result['predicted_return_percent']:.2f}%")
```

### 2. CSV 일괄 처리

```python
# CSV 파일로 대량 예측
results = predictor.predict_from_csv('input.csv', 'output.csv')
```

### 3. 성능 테스트

```bash
# 통합 테스트 실행
python tests/test_model.py
```

## 🛠️ 시스템 요구사항

### 최소 요구사항
- **Python**: 3.8+
- **GPU**: GTX 1080Ti (권장) 또는 동급
- **메모리**: 16GB+ RAM
- **CUDA**: 11.2+
- **저장공간**: 5GB+

### 핵심 의존성
```
tensorflow>=2.8.0,<2.15.0  # GPU 호환성
torch>=1.11.0               # BERT
transformers>=4.21.0        # Hugging Face
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## ⚠️ 주의사항

- **📢 이 모델의 예측은 참고용**이며 투자 조언이 아닙니다
- **💼 실제 투자 시 전문가와 상담**하시기 바랍니다
- **🔧 GTX 1080Ti 최적화**: 다른 GPU에서는 성능이 다를 수 있음

## 🏆 성능 최적화

- **61.3% 훈련 시간 단축** (8시간 → 3.1시간)
- **93.4% 연산량 절약** (동적 패딩)
- **배치 크기 최적화** (4 → 8)
- **GPU 메모리 효율성** 극대화

## 📈 기술적 세부사항

### 모델 아키텍처
- **텍스트 브랜치**: BERT(klue/bert-base) → Dense(64)
- **금융 브랜치**: LSTM(128) → Dense(64)
- **결합**: Concatenate → Dense(64) → Dropout → 출력

### GTX 1080Ti 최적화
- 동적 패딩으로 연산량 93.4% 절약
- Mixed Precision 비활성화 (특성 고려)
- 배치 크기 최적화로 61.3% 성능 향상

## 📄 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## 👥 기여

- **Issues**: 버그 리포트 및 기능 제안
- **Pull Requests**: 코드 개선 기여
- **Discussions**: 아이디어 공유

---

**🎯 GTX 1080Ti의 성능을 최대로 활용하여 금융 AI의 새로운 가능성을 탐험해보세요!**

*프로젝트 생성일: 2025-07-25*