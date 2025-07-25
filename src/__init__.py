"""
GTX 1080Ti 최적화 멀티모달 딥러닝 패키지

이 패키지는 금융 데이터와 뉴스 텍스트를 결합하여
주가 등락률을 예측하는 GTX 1080Ti 최적화 딥러닝 모델을 제공합니다.

공개 모듈:
- model: 메인 모델 아키텍처 및 훈련
- prediction: 예측 시스템
- training_monitor: 훈련 모니터링
"""

__version__ = '1.0.0'
__author__ = 'GTX 1080Ti Optimization Team'

# 공개 API
from .prediction import GTX1080TiPredictor

__all__ = [
    'GTX1080TiPredictor',
]