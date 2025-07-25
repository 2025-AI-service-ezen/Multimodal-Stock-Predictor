#!/usr/bin/env python3
"""
GTX 1080Ti 멀티모달 모델 빠른 시작 가이드

이 스크립트는 프로젝트의 기본 사용법을 보여줍니다.
초보자도 쉽게 따라할 수 있도록 단계별로 구성되었습니다.

실행 방법:
    python examples/quick_start.py

작성자: GTX 1080Ti 최적화 팀
업데이트: 2025-07-25
"""

import sys
import os
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import GTX1080TiPredictor


def main():
    """빠른 시작 가이드 메인 함수"""
    
    print("=" * 60)
    print("  GTX 1080Ti 멀티모달 모델 빠른 시작 가이드")
    print("  🚀 주가 등락률 예측 시스템")
    print("=" * 60)
    
    try:
        # ===== 1단계: 예측기 초기화 =====
        print("\n[1단계] 예측기 초기화...")
        predictor = GTX1080TiPredictor()
        
        # ===== 2단계: 환경 설정 =====
        print("\n[2단계] 환경 설정 및 모델 로드...")
        if not predictor.setup_environment():
            print("❌ 환경 설정 실패. 다음을 확인하세요:")
            print("  - 훈련된 모델 파일이 있는지 확인")
            print("  - GPU 드라이버가 설치되어 있는지 확인")
            print("  - 필요한 Python 패키지가 설치되어 있는지 확인")
            return
        
        # ===== 3단계: 샘플 데이터 생성 =====
        print("\n[3단계] 테스트용 샘플 데이터 생성...")
        
        # 데이터 폴더 생성 확인
        os.makedirs("./data/samples", exist_ok=True)
        
        # 샘플 데이터 생성 (예측기에 매소드가 있는 경우)
        try:
            sample_file = "./data/samples/quick_start_sample.csv"
            # 기본 샘플 데이터 생성
            print("ℹ️ 샘플 데이터를 직접 생성하거나 모델 데모를 실행해주세요.")
            print("📝 python src/prediction.py 를 실행하면 자동으로 데모가 실행됩니다.")
        except Exception as e:
            print(f"⚠️ 샘플 데이터 생성 실패: {e}")
        
        # ===== 4단계: 개별 예측 예시 =====
        print("\n[4단계] 개별 예측 수행...")
        print("-" * 40)
        
        # 예시 금융 데이터 (5일간)
        financial_data = np.array([
            [50000, 49500, 50500, 49000, 1000000, 0.01],   # 1일차
            [50500, 50000, 51000, 49800, 1100000, 0.01],   # 2일차  
            [51000, 50500, 51500, 50200, 1050000, 0.01],   # 3일차
            [51200, 51000, 51800, 50800, 980000, 0.004],   # 4일차
            [51500, 51200, 52000, 51000, 1200000, 0.006]   # 5일차
        ])
        
        # 예시 뉴스 텍스트
        news_text = "기업 실적 개선 전망 주가 상승 기대 투자자 관심 증가 매출 성장"
        
        print("📊 입력 데이터:")
        print(f"  💰 금융 데이터: 5일간 OHLCV + 전일비")
        print(f"  📰 뉴스 텍스트: '{news_text[:30]}...'")
        
        # 예측 수행
        try:
            result = predictor.predict(financial_data, news_text)
            
            if result:
                print(f"\n🎯 예측 결과:")
                print(f"  📈 예측 등락률: {result['predicted_return_percent']:+.2f}%")
                print(f"  🎭 방향성: {result['direction']}")
                print(f"  📊 신뢰도: {result['confidence']}")
                
                if 'interpretation' in result:
                    print(f"\n💡 해석:")
                    for interpretation in result['interpretation']:
                        print(f"  • {interpretation}")
            else:
                print("❌ 예측 실패")
        except Exception as e:
            print(f"❌ 예측 오류: {e}")
        
        print("\n🎉 빠른 시작 가이드 완료!")
        print("📚 더 자세한 사용법은 README.md를 참고하세요.")
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        print("🔧 다음을 확인해주세요:")
        print("  1. 모든 의존성이 설치되었는지 (pip install -r requirements.txt)")
        print("  2. GPU 드라이버가 올바르게 설치되었는지")
        print("  3. Python 버전이 3.8 이상인지")


if __name__ == "__main__":
    main()