"""
GTX 1080Ti 멀티모달 모델 예측 시스템

이 모듈은 훈련된 GTX 1080Ti 최적화 멀티모달 모델을 사용하여
새로운 금융 데이터와 뉴스 텍스트로 주가 등락률을 예측하는 시스템입니다.

주요 기능:
1. 새로운 금융 데이터 입력 및 전처리
2. 뉴스 텍스트 BERT 토큰화
3. 훈련된 모델로 등락률 예측
4. 예측 결과 해석 및 신뢰도 분석
5. CSV 파일 일괄 처리
6. 샘플 데이터 자동 생성

작성자: GTX 1080Ti 최적화 팀
최종 업데이트: 2025-07-25
버전: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, TFBertModel
import warnings
warnings.filterwarnings('ignore')


class GTX1080TiPredictor:
    """
    GTX 1080Ti 최적화 멀티모달 모델 예측 클래스
    
    이 클래스는 훈련된 BERT+LSTM 멀티모달 모델을 사용하여
    금융 데이터와 뉴스 텍스트로부터 주가 등락률을 예측합니다.
    """
    
    def __init__(self, base_path="./"):
        """예측기 초기화"""
        self.base_path = base_path
        self.model = None
        self.scaler = None
        self.tokenizer = None
        
        # BERT 모델 설정
        self.BERT_MODEL_NAME = 'klue/bert-base'
        self.MAX_LEN = 256
        
        # 파일 경로 설정
        self.scaler_path = os.path.join(base_path, "data/processed/financial_data_scaler.pkl")
        self.model_path = os.path.join(base_path, "models/trained/multimodal_finetuned_model_best.h5")
        self.model_path_alt = os.path.join(base_path, "models/trained/multimodal_finetuned_model.h5")
        
        print("🎯 GTX 1080Ti 멀티모달 예측 시스템")
        print("📊 금융 데이터 + 뉴스 텍스트 → 주가 등락률 예측")
        print("=" * 60)
    
    def setup_environment(self):
        """예측 환경 설정 및 모델 로드"""
        print("\n[환경 설정] GPU 및 모델 초기화...")
        
        # 1. GPU 설정
        self._setup_gpu()
        
        # 2. 스케일러 로드
        if not self._load_scaler():
            return False
        
        # 3. 토크나이저 로드
        if not self._load_tokenizer():
            return False
        
        # 4. 모델 로드
        if not self._load_model():
            print("⚠️ 훈련된 모델을 로드할 수 없습니다.")
            return False
        
        print("\n✅ 예측 환경 설정 완료")
        return True
    
    def _setup_gpu(self):
        """GTX 1080Ti에 최적화된 GPU 설정"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ GPU 활성화: {len(gpus)}개 디바이스")
                
                for i, gpu in enumerate(gpus):
                    details = tf.config.experimental.get_device_details(gpu)
                    device_name = details.get('device_name', 'Unknown')
                    print(f"   GPU {i}: {device_name}")
            else:
                print("⚠️ GPU를 찾을 수 없습니다. CPU로 실행됩니다.")
                
        except RuntimeError as e:
            print(f"⚠️ GPU 설정 중 오류: {e}")
    
    def _load_scaler(self):
        """금융 데이터 정규화용 스케일러 로드"""
        try:
            if not os.path.exists(self.scaler_path):
                print(f"❌ 스케일러 파일을 찾을 수 없습니다: {self.scaler_path}")
                return False
            
            self.scaler = joblib.load(self.scaler_path)
            print(f"✅ 스케일러 로드 성공: {self.scaler_path}")
            print(f"   정규화 범위: [{self.scaler.feature_range[0]}, {self.scaler.feature_range[1]}]")
            
            return True
            
        except Exception as e:
            print(f"❌ 스케일러 로드 실패: {e}")
            return False
    
    def _load_tokenizer(self):
        """BERT 토크나이저 로드"""
        try:
            print(f"📝 BERT 토크나이저 로드 중: {self.BERT_MODEL_NAME}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL_NAME)
            
            print("✅ BERT 토크나이저 로드 성공")
            print(f"   어휘 크기: {len(self.tokenizer):,}개")
            print(f"   최대 길이: {self.MAX_LEN} 토큰")
            
            return True
            
        except Exception as e:
            print(f"❌ 토크나이저 로드 실패: {e}")
            return False
    
    def _load_model(self):
        """훈련된 모델 로드"""
        try:
            # 최적 모델 우선 시도
            if os.path.exists(self.model_path):
                print(f"📁 최적 모델 로드 시도: {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
            elif os.path.exists(self.model_path_alt):
                print(f"📁 대체 모델 로드 시도: {self.model_path_alt}")
                self.model = tf.keras.models.load_model(self.model_path_alt)
            else:
                print("❌ 모델 파일을 찾을 수 없습니다.")
                return False
            
            print("✅ 최적 모델 로드 성공")
            print(f"   모델 파라미터: {self.model.count_params():,}개")
            print(f"   입력 형태: {[input.shape for input in self.model.inputs]}")
            print(f"   출력 형태: {self.model.output.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def predict(self, financial_data, news_text):
        """
        단일 샘플 예측
        
        Args:
            financial_data (np.array): 5일간 금융 데이터 (5 × 6)
            news_text (str): 뉴스 텍스트
            
        Returns:
            dict: 예측 결과
        """
        try:
            print("\n[예측 수행] 새로운 데이터로 주가 등락률 예측...")
            
            # 1. 금융 데이터 전처리
            print("📊 금융 데이터 전처리 중...")
            financial_processed = self._preprocess_financial_data(financial_data)
            
            # 2. 텍스트 전처리
            print("📰 뉴스 텍스트 전처리 중...")
            text_processed = self._preprocess_text_data(news_text)
            
            # 3. 예측 수행
            print("🔮 모델 예측 수행 중...")
            prediction = self.model.predict([
                financial_processed['data'],
                text_processed['input_ids'],
                text_processed['attention_mask'],
                text_processed['token_type_ids']
            ], verbose=0)
            
            # 4. 결과 해석
            predicted_return = float(prediction[0][0])
            predicted_return_percent = predicted_return * 100
            
            # 방향성 판단
            if predicted_return_percent > 1.0:
                direction = "상승"
            elif predicted_return_percent < -1.0:
                direction = "하락"
            else:
                direction = "횡보"
            
            # 신뢰도 계산 (절댓값 기준)
            abs_return = abs(predicted_return_percent)
            if abs_return > 3.0:
                confidence = "높음"
            elif abs_return > 1.0:
                confidence = "보통"
            else:
                confidence = "낮음"
            
            print(f"✅ 예측 완료: {predicted_return_percent:.2f}% ({direction})")
            
            return {
                'predicted_return': predicted_return,
                'predicted_return_percent': predicted_return_percent,
                'direction': direction,
                'confidence': confidence,
                'interpretation': self._generate_interpretation(
                    predicted_return_percent, direction, confidence
                )
            }
            
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None
    
    def _preprocess_financial_data(self, financial_data):
        """금융 데이터 전처리"""
        try:
            # 형태 확인
            if financial_data.shape != (5, 6):
                raise ValueError(f"금융 데이터 형태가 올바르지 않습니다: {financial_data.shape}, 예상: (5, 6)")
            
            # 배치 차원 추가
            financial_batch = financial_data.reshape(1, 5, 6)
            
            # 정규화 (reshape for scaler)
            original_shape = financial_batch.shape
            reshaped = financial_batch.reshape(-1, financial_batch.shape[-1])
            normalized = self.scaler.transform(reshaped)
            normalized = normalized.reshape(original_shape)
            
            print(f"✅ 금융 데이터 전처리 완료: {normalized.shape}")
            print(f"   정규화 범위: [{normalized.min():.3f}, {normalized.max():.3f}]")
            
            return {'data': normalized}
            
        except Exception as e:
            print(f"❌ 금융 데이터 전처리 실패: {e}")
            return None
    
    def _preprocess_text_data(self, news_text):
        """뉴스 텍스트 전처리"""
        try:
            # BERT 토큰화
            encoding = self.tokenizer(
                news_text,
                truncation=True,
                padding='max_length',
                max_length=self.MAX_LEN,
                return_tensors='tf'
            )
            
            # 실제 토큰 길이 계산
            actual_tokens = len([t for t in self.tokenizer.tokenize(news_text) if t != '[PAD]'])
            
            print(f"✅ 텍스트 전처리 완료:")
            print(f"   원본 길이: {len(news_text)}자")
            print(f"   실제 토큰: {actual_tokens}개")
            print(f"   패딩 후: {self.MAX_LEN}개")
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids']
            }
            
        except Exception as e:
            print(f"❌ 텍스트 전처리 실패: {e}")
            return None
    
    def _generate_interpretation(self, predicted_return, direction, confidence):
        """예측 결과 해석 생성"""
        interpretations = []
        
        # 기본 해석
        if direction == "상승":
            interpretations.append(f"모델이 {predicted_return:.2f}% 상승을 예측했습니다.")
        elif direction == "하락":
            interpretations.append(f"모델이 {predicted_return:.2f}% 하락을 예측했습니다.")
        else:
            interpretations.append(f"모델이 {predicted_return:.2f}% 소폭 변동을 예측했습니다.")
        
        # 신뢰도 해석
        if confidence == "높음":
            interpretations.append("예측 신뢰도가 높습니다.")
        elif confidence == "보통":
            interpretations.append("예측 신뢰도가 보통입니다.")
        else:
            interpretations.append("예측 신뢰도가 낮으므로 신중한 판단이 필요합니다.")
        
        # 투자 주의사항
        interpretations.append("이 예측은 참고용이며 실제 투자 결정 시 전문가와 상담하세요.")
        
        return interpretations
    
    def create_sample_input(self, output_path="sample_input.csv", num_samples=3):
        """테스트용 샘플 데이터 생성"""
        try:
            print("\n[샘플 생성] 테스트용 샘플 데이터 생성...")
            print(f"📁 저장 경로: {output_path}")
            print(f"📊 샘플 수: {num_samples}개")
            
            # 디렉터리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            samples = []
            scenarios = ["상승", "하락", "횡보"]
            
            for i in range(num_samples):
                scenario = scenarios[i % len(scenarios)]
                
                # 시나리오별 데이터 생성
                if scenario == "상승":
                    # 상승 패턴 (5일간 점진적 상승)
                    base_price = 50000
                    prices = [base_price + j * 200 + np.random.randint(-100, 100) for j in range(5)]
                    volumes = [1000000 + np.random.randint(-100000, 200000) for _ in range(5)]
                    changes = [0.004 + np.random.uniform(-0.002, 0.008) for _ in range(5)]
                    news = "기업 실적 개선 전망 주가 상승 기대 투자자 관심 증가"
                    
                elif scenario == "하락":
                    # 하락 패턴 (5일간 점진적 하락)
                    base_price = 52000
                    prices = [base_price - j * 300 + np.random.randint(-50, 50) for j in range(5)]
                    volumes = [1100000 + np.random.randint(-50000, 100000) for _ in range(5)]
                    changes = [-0.006 + np.random.uniform(-0.004, 0.002) for _ in range(5)]
                    news = "업계 전망 부정적 실적 우려 증가 매도 압력"
                    
                else:  # 횡보
                    # 횡보 패턴 (작은 변동)
                    base_price = 51000
                    prices = [base_price + np.random.randint(-200, 200) for _ in range(5)]
                    volumes = [1050000 + np.random.randint(-100000, 100000) for _ in range(5)]
                    changes = [np.random.uniform(-0.003, 0.003) for _ in range(5)]
                    news = "시장 관망세 지속 거래량 감소 횡보 전망"
                
                # CSV 행 생성
                row = {}
                trend_5days = ((prices[-1] - prices[0]) / prices[0]) * 100
                
                for day in range(1, 6):
                    price = prices[day-1]
                    # OHLC 생성 (Close 기준으로 변동)
                    open_price = price + np.random.randint(-100, 100)
                    high_price = max(price, open_price) + np.random.randint(0, 150)
                    low_price = min(price, open_price) - np.random.randint(0, 150)
                    
                    row[f'day{day}_close'] = price
                    row[f'day{day}_open'] = open_price
                    row[f'day{day}_high'] = high_price
                    row[f'day{day}_low'] = low_price
                    row[f'day{day}_volume'] = volumes[day-1]
                    row[f'day{day}_change'] = changes[day-1]
                
                row['news_text'] = news
                row['scenario'] = f"{scenario} 시나리오 (5일 추세: {trend_5days:+.1f}%)"
                
                samples.append(row)
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(samples)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print("✅ 샘플 파일 생성 완료")
            print("📋 파일 형식:")
            print("   - day1~5_close/open/high/low/volume/change: 5일간 금융 데이터")
            print("   - news_text: 관련 뉴스 텍스트")
            print("   - scenario: 시나리오 설명 (참고용)")
            
            print(f"\n📊 생성된 샘플 미리보기:")
            for i, sample in enumerate(samples):
                print(f"   샘플 {i+1}: {sample['scenario']}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ 샘플 파일 생성 실패: {e}")
            return None
    
    def predict_from_csv(self, input_path, output_path):
        """CSV 파일 일괄 예측"""
        try:
            print(f"\n[일괄 예측] CSV 파일 처리 시작...")
            print(f"📁 입력 파일: {input_path}")
            
            # 데이터 로드
            df = pd.read_csv(input_path)
            print(f"✅ 데이터 로드 완료: {len(df)}개 샘플")
            
            results = []
            print("🔮 예측 진행 중...")
            
            for idx, row in df.iterrows():
                try:
                    # 금융 데이터 추출
                    financial_data = np.array([
                        [row['day1_close'], row['day1_open'], row['day1_high'], 
                         row['day1_low'], row['day1_volume'], row['day1_change']],
                        [row['day2_close'], row['day2_open'], row['day2_high'], 
                         row['day2_low'], row['day2_volume'], row['day2_change']],
                        [row['day3_close'], row['day3_open'], row['day3_high'], 
                         row['day3_low'], row['day3_volume'], row['day3_change']],
                        [row['day4_close'], row['day4_open'], row['day4_high'], 
                         row['day4_low'], row['day4_volume'], row['day4_change']],
                        [row['day5_close'], row['day5_open'], row['day5_high'], 
                         row['day5_low'], row['day5_volume'], row['day5_change']]
                    ])
                    
                    news_text = str(row['news_text'])
                    
                    # 예측 수행
                    result = self.predict(financial_data, news_text)
                    
                    if result:
                        # 결과 저장
                        result_row = {
                            'sample_index': idx,
                            'predicted_return': result['predicted_return'],
                            'predicted_return_percent': result['predicted_return_percent'],
                            'direction': result['direction'],
                            'confidence': result['confidence']
                        }
                        
                        # 추가 분석 정보
                        trend_5days = ((financial_data[-1][0] - financial_data[0][0]) / financial_data[0][0]) * 100
                        volatility = np.std([day[0] for day in financial_data]) / np.mean([day[0] for day in financial_data])
                        
                        result_row['price_trend_5days'] = trend_5days
                        result_row['volatility'] = volatility
                        result_row['has_news'] = len(news_text.strip()) > 0
                        
                        results.append(result_row)
                        
                    else:
                        # 예측 실패
                        results.append({
                            'sample_index': idx,
                            'predicted_return': None,
                            'predicted_return_percent': None,
                            'direction': 'ERROR',
                            'confidence': 'ERROR'
                        })
                        
                except Exception as e:
                    print(f"⚠️ 샘플 {idx} 예측 실패: {e}")
                    results.append({
                        'sample_index': idx,
                        'predicted_return': None,
                        'predicted_return_percent': None,
                        'direction': 'ERROR',
                        'confidence': 'ERROR'
                    })
                
                # 진행률 출력
                if (idx + 1) % max(1, len(df) // 10) == 0 or idx == len(df) - 1:
                    print(f"   진행률: {idx + 1}/{len(df)} ({(idx + 1)/len(df)*100:.1f}%)")
            
            # 결과 저장
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            # 결과 요약
            valid_results = results_df[results_df['direction'] != 'ERROR']
            success_rate = len(valid_results) / len(results_df) * 100
            
            print(f"\n📊 예측 결과 요약:")
            print(f"   성공: {len(valid_results)}개")
            print(f"   실패: {len(results_df) - len(valid_results)}개")
            print(f"   성공률: {success_rate:.1f}%")
            
            if len(valid_results) > 0:
                direction_counts = valid_results['direction'].value_counts().to_dict()
                avg_return = valid_results['predicted_return_percent'].mean()
                print(f"   방향 분포: {direction_counts}")
                print(f"   평균 예측 등락률: {avg_return:.2f}%")
            
            print(f"✅ 예측 결과 저장: {output_path}")
            
            return results_df
            
        except Exception as e:
            print(f"❌ 일괄 예측 실패: {e}")
            return None


def main():
    """예측 시스템 데모 실행"""
    print("=" * 60)
    print("  GTX 1080Ti 멀티모달 예측 시스템")
    print("  주가 등락률 예측 (BERT + LSTM)")
    print("=" * 60)
    
    # 예측기 초기화
    predictor = GTX1080TiPredictor()
    
    # 환경 설정
    if not predictor.setup_environment():
        print("❌ 환경 설정에 실패했습니다.")
        return
    
    print("\n🎯 예측 시스템 사용법:")
    print("━" * 42)
    print("1️⃣ 단일 예측:")
    print("   predictor.predict(financial_data, news_text)")
    print("   ")
    print("2️⃣ 일괄 예측:")
    print("   predictor.predict_from_csv('input.csv', 'output.csv')")
    print("   ")
    print("3️⃣ 샘플 생성:")
    print("   predictor.create_sample_input('sample.csv')")
    
    # 데모 실행
    print(f"\n[데모] 샘플 데이터 생성 및 테스트 예측...")
    
    # 샘플 데이터 생성
    sample_path = predictor.create_sample_input("data/samples/demo_input.csv")
    
    if sample_path:
        print(f"\n[데모] 샘플 데이터로 예측 테스트...")
        
        # 일괄 예측 수행
        results_df = predictor.predict_from_csv(
            sample_path,
            "data/samples/demo_results.csv"
        )
        
        if results_df is not None:
            # 결과 요약 출력
            valid_results = results_df[results_df['direction'] != 'ERROR']
            
            print(f"\n📊 === 예측 결과 요약 ===")
            print(f"✅ 성공한 예측: {len(valid_results)}개")
            
            if len(valid_results) > 0:
                print(f"\n📈 예측 결과 상세:")
                print("-" * 50)
                
                for _, row in valid_results.iterrows():
                    direction_emoji = "📈" if row['direction'] == "상승" else "📉" if row['direction'] == "하락" else "🟡"
                    print(f"{direction_emoji} 샘플 {int(row['sample_index'])+1}: {row['direction']} {row['predicted_return_percent']:+.2f}% (신뢰도: {row['confidence']})")
                
                print(f"\n📊 통계 요약:")
                direction_counts = valid_results['direction'].value_counts().to_dict()
                avg_return = valid_results['predicted_return_percent'].mean()
                print(f"   평균 예측 등락률: {avg_return:+.2f}%")
                print(f"   방향 분포: {direction_counts}")
    
    print(f"\n💡 === 사용 가이드 ===")
    print(f"📁 생성된 파일:")
    print(f"   - 입력 샘플: data/samples/demo_input.csv")
    print(f"   - 예측 결과: data/samples/demo_results.csv")
    print(f"   ")
    print(f"🔧 직접 사용하려면:")
    print(f"   1. CSV 파일을 생성하고 필수 컬럼을 포함시키세요")
    print(f"   2. predict_from_csv() 함수로 일괄 예측하세요")
    print(f"   3. 또는 predict() 함수로 개별 예측하세요")
    print(f"   ")
    print(f"⚠️ 주의사항:")
    print(f"   - 이 예측은 참고용이며 투자 조언이 아닙니다")
    print(f"   - 실제 투자 시 전문가와 상담하세요")
    print(f"   - 모델의 한계와 시장 불확실성을 고려하세요")
    
    print("=" * 60)
    print("  예측 시스템 데모 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()