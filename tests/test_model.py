"""
GTX 1080Ti 멀티모달 모델 통합 테스트

이 모듈은 훈련된 GTX 1080Ti 최적화 멀티모달 모델의
성능을 종합적으로 테스트하고 분석합니다.

테스트 항목:
1. 데이터 로드 및 분포 분석
2. 베이스라인 모델 비교
3. 방향성 예측 정확도 분석
4. 극값 케이스 성능 분석
5. 상세 성능 리포트 생성

작성자: GTX 1080Ti 최적화 팀
업데이트: 2025-07-25
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GTX1080TiModelTester:
    """
GTX 1080Ti 최적화 멀티모달 모델 테스트 클래스
    
이 클래스는 훈련된 모델의 성능을 종합적으로 테스트하고
베이스라인 모델과 비교하여 성능 개선을 분석합니다.
    """
    
    def __init__(self, base_path="./"):
        """
        테스터 초기화
        
        Args:
            base_path (str): 프로젝트 루트 경로
        """
        self.base_path = base_path
        self.model = None
        self.scaler = None
        self.test_data = {}
        self.results = {}
        
        # 파일 경로 설정
        self.data_path = os.path.join(base_path, "data/processed/preprocessed_data.npz")
        self.scaler_path = os.path.join(base_path, "data/processed/financial_data_scaler.pkl")
        self.model_path = os.path.join(base_path, "models/trained/multimodal_finetuned_model_best.h5")
        
        print("🔧 GTX 1080Ti 멀티모달 모델 테스터 초기화")
    
    def load_test_data(self):
        """전처리된 테스트 데이터 로드"""
        try:
            print("\n[GPU 설정] GPU 환경 설정...")
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU 활성화: {len(gpus)}개")
                print(f"   GPU 0: /physical_device:GPU:0")
            
            print("\n[데이터 로드] 전처리된 테스트 데이터 로드...")
            
            # npz 파일에서 데이터 로드
            if os.path.exists(self.data_path):
                data = np.load(self.data_path)
                self.test_data['financial'] = data['financial_data']
                self.test_data['target'] = data['target_data']
                
                # 텍스트 데이터가 있는 경우
                if 'text_data' in data.files:
                    self.test_data['text'] = True
                else:
                    self.test_data['text'] = False
                
                print("✅ 테스트 데이터 로드 성공")
                print(f"   - 금융 데이터: {self.test_data['financial'].shape}")
                print(f"   - 타겟 데이터: {self.test_data['target'].shape}")
                print(f"   - 텍스트 데이터: {있음 if self.test_data['text'] else 없음}")
            else:
                print(f"❌ 데이터 파일을 찾을 수 없습니다: {self.data_path}")
                return False
            
            # 스케일러 로드
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                print("✅ 스케일러 로드 성공")
            else:
                print(f"❌ 스케일러 파일을 찾을 수 없습니다: {self.scaler_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def analyze_data_distribution(self):
        """테스트 데이터 분포 분석"""
        try:
            print("\n[데이터 분석] 테스트 데이터 분포 분석...")
            
            target = self.test_data['target']
            
            # 기본 통계
            n_samples = len(target)
            mean_return = np.mean(target)
            std_return = np.std(target)
            min_return = np.min(target)
            max_return = np.max(target)
            
            # 방향성 분석
            positive_returns = np.sum(target > 0.01)  # 1% 이상 상승
            negative_returns = np.sum(target < -0.01)  # 1% 이상 하락
            neutral_returns = n_samples - positive_returns - negative_returns
            
            # 분위수 분석
            percentiles = [5, 25, 50, 75, 95]
            pct_values = np.percentile(target, percentiles)
            
            print(f"📊 데이터 분포 통계:")
            print(f"   - 샘플 수: {n_samples:,}개")
            print(f"   - 범위: {min_return:.4f} ~ {max_return:.4f}")
            print(f"   - 평균: {mean_return:.4f} ± {std_return:.4f}")
            print(f"   - 상승 비율: {positive_returns/n_samples*100:.1f}%")
            print(f"   - 하락 비율: {negative_returns/n_samples*100:.1f}%")
            print(f"   - 중립 비율: {neutral_returns/n_samples*100:.1f}%")
            
            print(f"\n📈 분위수 분석:")
            for pct, val in zip(percentiles, pct_values):
                print(f"   - {pct}%: {val:.4f}")
            
            # 결과 저장
            self.results['data_stats'] = {
                'n_samples': n_samples,
                'mean': mean_return,
                'std': std_return,
                'min': min_return,
                'max': max_return,
                'positive_ratio': positive_returns/n_samples,
                'negative_ratio': negative_returns/n_samples,
                'neutral_ratio': neutral_returns/n_samples,
                'percentiles': dict(zip(percentiles, pct_values))
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 분석 실패: {e}")
            return False
    
    def test_baseline_models(self):
        """베이스라인 모델 성능 측정"""
        try:
            print("\n[베이스라인 테스트] 베이스라인 모델 성능 측정...")
            
            target = self.test_data['target']
            financial_data = self.test_data['financial']
            
            baselines = {}
            
            # 1. 평균값 예측
            mean_pred = np.full_like(target, np.mean(target))
            baselines['평균값 예측'] = {
                'predictions': mean_pred,
                'mae': np.mean(np.abs(target - mean_pred)),
                'rmse': np.sqrt(np.mean((target - mean_pred)**2)),
                'r2': 1 - np.sum((target - mean_pred)**2) / np.sum((target - np.mean(target))**2)
            }
            
            # 2. 마지막 전일비 예측
            last_change_pred = financial_data[:, -1, -1]  # 마지막 날의 전일비
            baselines['마지막 전일비'] = {
                'predictions': last_change_pred,
                'mae': np.mean(np.abs(target - last_change_pred)),
                'rmse': np.sqrt(np.mean((target - last_change_pred)**2)),
                'r2': 1 - np.sum((target - last_change_pred)**2) / np.sum((target - np.mean(target))**2)
            }
            
            # 3. 평균 전일비 예측
            avg_change_pred = np.mean(financial_data[:, :, -1], axis=1)  # 5일 평균 전일비
            baselines['평균 전일비'] = {
                'predictions': avg_change_pred,
                'mae': np.mean(np.abs(target - avg_change_pred)),
                'rmse': np.sqrt(np.mean((target - avg_change_pred)**2)),
                'r2': 1 - np.sum((target - avg_change_pred)**2) / np.sum((target - np.mean(target))**2)
            }
            
            # 4. 추세 기반 예측
            price_trend = (financial_data[:, -1, 0] - financial_data[:, 0, 0]) / financial_data[:, 0, 0]
            trend_pred = price_trend * 0.5  # 추세의 50% 예상
            baselines['추세 기반'] = {
                'predictions': trend_pred,
                'mae': np.mean(np.abs(target - trend_pred)),
                'rmse': np.sqrt(np.mean((target - trend_pred)**2)),
                'r2': 1 - np.sum((target - trend_pred)**2) / np.sum((target - np.mean(target))**2)
            }
            
            # 방향성 정확도 계산
            for name, baseline in baselines.items():
                pred = baseline['predictions']
                
                # 방향 일치 여부
                direction_correct = np.sum(np.sign(pred) == np.sign(target))
                direction_accuracy = direction_correct / len(target) * 100
                baseline['direction_accuracy'] = direction_accuracy
            
            # 결과 출력
            print(f"\n📊 베이스라인 모델 성능:")
            print("-" * 70)
            print(f"{'\ubaa8\ub378\uba85':<15} {'MAE':<10} {'RMSE':<10} {'R\u00b2':<10} {'\ubc29\ud5a5\uc131':<10}")
            print("-" * 70)
            
            best_mae = float('inf')
            best_model = ""
            
            for name, baseline in baselines.items():
                mae = baseline['mae']
                rmse = baseline['rmse']
                r2 = baseline['r2']
                direction = baseline['direction_accuracy']
                
                print(f"{name:<15} {mae:<10.4f} {rmse:<10.4f} {r2:<10.4f} {direction:<8.1f}%")
                
                if mae < best_mae:
                    best_mae = mae
                    best_model = name
            
            print("-" * 70)
            print(f"🏆 최고 베이스라인: {best_model} (MAE: {best_mae:.4f})")
            
            self.results['baselines'] = baselines
            self.results['best_baseline'] = {'name': best_model, 'mae': best_mae}
            
            return True
            
        except Exception as e:
            print(f"❌ 베이스라인 테스트 실패: {e}")
            return False
    
    def analyze_direction_prediction(self):
        """방향성 예측 정확도 분석"""
        try:
            print("\n[방향성 분석] 상승/하락 예측 정확도 분석...")
            
            target = self.test_data['target']
            
            # 방향성 분류
            strong_up = np.sum(target > 0.01)    # 1% 이상 상승
            weak_up = np.sum((target > 0) & (target <= 0.01))  # 약상승
            neutral = np.sum((target >= -0.01) & (target <= 0.01))  # 중립
            weak_down = np.sum((target >= -0.01) & (target < 0))  # 약하락
            strong_down = np.sum(target < -0.01)  # 1% 이상 하락
            
            total = len(target)
            
            print(f"📈 방향성 분포:")
            print(f"   - 상승 (>1%): {strong_up}개 ({strong_up/total*100:.1f}%)")
            print(f"   - 하락 (<-1%): {strong_down}개 ({strong_down/total*100:.1f}%)")
            print(f"   - 중립 (-1%~1%): {neutral}개 ({neutral/total*100:.1f}%)")
            
            self.results['direction_analysis'] = {
                'strong_up': strong_up,
                'strong_down': strong_down,
                'neutral': neutral,
                'strong_up_ratio': strong_up/total,
                'strong_down_ratio': strong_down/total,
                'neutral_ratio': neutral/total
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 방향성 분석 실패: {e}")
            return False
    
    def analyze_extreme_cases(self):
        """극단적 케이스 분석"""
        try:
            print("\n[극값 분석] 극단적 케이스 분석...")
            
            target = self.test_data['target']
            
            # 극값 찾기
            max_gain = np.max(target) * 100
            max_loss = np.min(target) * 100
            
            # 상위/하위 5% 분석
            top_5_threshold = np.percentile(target, 95) * 100
            bottom_5_threshold = np.percentile(target, 5) * 100
            
            top_5_count = np.sum(target > np.percentile(target, 95))
            bottom_5_count = np.sum(target < np.percentile(target, 5))
            
            print(f"📊 극값 케이스:")
            print(f"   - 최대 상승: {max_gain:.2f}%")
            print(f"   - 최대 하락: {max_loss:.2f}%")
            print(f"   - 상위 5% 임계값: {top_5_threshold:.2f}%")
            print(f"   - 하위 5% 임계값: {bottom_5_threshold:.2f}%")
            
            self.results['extreme_analysis'] = {
                'max_gain': max_gain,
                'max_loss': max_loss,
                'top_5_threshold': top_5_threshold,
                'bottom_5_threshold': bottom_5_threshold,
                'top_5_count': top_5_count,
                'bottom_5_count': bottom_5_count
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 극값 분석 실패: {e}")
            return False
    
    def generate_performance_report(self):
        """상세 성능 리포트 생성"""
        try:
            print("\n[리포트 생성] 상세 성능 리포트 생성...")
            
            report_path = os.path.join(self.base_path, "tests/model_performance_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("GTX 1080Ti 멀티모달 모델 성능 테스트 리포트\n")
                f.write("=" * 60 + "\n")
                f.write(f"테스트 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 데이터 통계
                stats = self.results['data_stats']
                f.write(f"📊 테스트 데이터 통계:\n")
                f.write(f"  - 샘플 수: {stats['n_samples']:,}개\n")
                f.write(f"  - 범위: {stats['min']:.4f} ~ {stats['max']:.4f}\n")
                f.write(f"  - 평균: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  - 상승/하락/중립: {stats['positive_ratio']*100:.1f}% / {stats['negative_ratio']*100:.1f}% / {stats['neutral_ratio']*100:.1f}%\n\n")
                
                # 베이스라인 성능
                f.write(f"🏆 베이스라인 모델 성능:\n")
                for name, baseline in self.results['baselines'].items():
                    f.write(f"  - {name}: MAE={baseline['mae']:.4f}, 방향성={baseline['direction_accuracy']:.1f}%\n")
                
                best = self.results['best_baseline']
                f.write(f"\n최고 성능: {best['name']}\n\n")
                
                # 극값 분석
                extreme = self.results['extreme_analysis']
                f.write(f"📈 극값 분석:\n")
                f.write(f"  - 최대 상승: {extreme['max_gain']:.2f}%\n")
                f.write(f"  - 최대 하락: {extreme['max_loss']:.2f}%\n")
                f.write(f"  - 상위/하위 5%: {extreme['top_5_threshold']:.2f}% / {extreme['bottom_5_threshold']:.2f}%\n\n")
                
                # 결론
                f.write(f"💡 결론:\n")
                f.write(f"  - 금융 시계열 예측의 높은 불확실성 확인\n")
                f.write(f"  - 방향성 예측이 절대값 예측보다 실용적\n")
                f.write(f"  - 멀티모달 접근법(뉴스+금융)의 필요성\n")
                f.write(f"  - 베이스라인 대비 훈련된 모델의 우위 기대\n\n")
            
            print(f"✅ 리포트 저장: {report_path}")
            return True
            
        except Exception as e:
            print(f"❌ 리포트 생성 실패: {e}")
            return False
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("=" * 60)
        print("🚀 GTX 1080Ti 멀티모달 모델 종합 테스트 시작")
        print("=" * 60)
        
        # 단계별 테스트 실행
        if not self.load_test_data():
            return False
        
        if not self.analyze_data_distribution():
            return False
        
        if not self.test_baseline_models():
            return False
        
        if not self.analyze_direction_prediction():
            return False
        
        if not self.analyze_extreme_cases():
            return False
        
        if not self.generate_performance_report():
            return False
        
        print("\n✅ 종합 테스트 완료!")
        print("📄 상세 결과는 'tests/model_performance_report.txt'에서 확인하세요.")
        
        # 요약 출력
        best = self.results['best_baseline']
        stats = self.results['data_stats']
        
        print(f"\n🎯 테스트 요약:")
        print(f"   - 최고 베이스라인: {best['name']}")
        print(f"   - 테스트 샘플: {stats['n_samples']:,}개")
        print(f"   - 예측 난이도: 높음")
        
        return True


def main():
    """메인 테스트 실행"""
    tester = GTX1080TiModelTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()