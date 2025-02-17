"""
機械学習モデルの評価スクリプト
学習したモデルの性能評価と最適なモデルの選択を行う
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from ml_strategy_patterns import PredictionTargets
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from p19_model_training import preprocess_data

class ModelEvaluator:
    def __init__(self, results_path: str = './results/training_results.json'):
        """
        Parameters:
        -----------
        results_path : str
            学習結果のJSONファイルパス
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.models_path = './models'
        self.evaluation_metrics = {}
    
    def load_model(self, model_name: str) -> Tuple[object, Dict]:
        """モデルと設定の読み込み"""
        base_path = f"{self.models_path}/{model_name}"
        
        # 設定の読み込み
        with open(f"{base_path}_config.json", 'r') as f:
            config = json.load(f)
        
        # モデルの読み込み
        if config['model_type'] == 'lstm':
            model = tf.keras.models.load_model(f"{base_path}_model.keras")
        else:
            model = joblib.load(f"{base_path}_model.joblib")
        
        return model, config
    
    def get_target_columns(self, df: pd.DataFrame, target: str) -> np.ndarray:
        """One-Hotエンコーディング後のターゲット変数を取得"""
        target_cols = [col for col in df.columns if col.startswith(f"{target}_")]
        if target_cols:
            return np.argmax(df[target_cols].values, axis=1)
        elif target in df.columns:
            return df[target].values
        else:
            raise ValueError(f"予測対象 {target} が見つかりません")
    
    def prepare_sequence_data(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """LSTMのための時系列データ準備"""
        X_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:(i + sequence_length)])
        return np.array(X_seq)
    
    def evaluate_classification_models(self, df: pd.DataFrame) -> Dict:
        """分類モデルの評価"""
        classification_metrics = {}
        
        for model_name, result in self.results.items():
            if result['metrics'].get('accuracy') is not None:  # 分類モデルの場合
                try:
                    model, config = self.load_model(model_name)
                    
                    # 特徴量とターゲットの準備
                    X = df[config['features']].values
                    y = self.get_target_columns(df, config['prediction_target'])
                    
                    # スケーラーの読み込みと適用
                    feature_scaler = joblib.load(f"{self.models_path}/{model_name}_feature_scaler.joblib")
                    X = feature_scaler.transform(X)
                    
                    # LSTMの場合、シーケンスデータを準備
                    if config['model_type'] == 'lstm':
                        X = self.prepare_sequence_data(X)
                        y = y[10:]  # シーケンス長分のターゲットをスキップ
                    
                    # 予測
                    y_pred = model.predict(X)
                    if config['model_type'] == 'lstm':
                        y_pred = (y_pred > 0.5).astype(int)
                    
                    # 評価指標の計算
                    metrics = {
                        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                        'classification_report': classification_report(y, y_pred, output_dict=True),
                        'feature_importance': self.get_feature_importance(model, config['features'])
                    }
                    
                    classification_metrics[model_name] = metrics
                except Exception as e:
                    print(f"モデル {model_name} の評価中にエラーが発生しました: {str(e)}")
                    continue
        
        return classification_metrics
    
    def evaluate_regression_models(self, df: pd.DataFrame) -> Dict:
        """回帰モデルの評価"""
        regression_metrics = {}
        
        for model_name, result in self.results.items():
            if result['metrics'].get('mse') is not None:  # 回帰モデルの場合
                try:
                    model, config = self.load_model(model_name)
                    
                    # 特徴量とターゲットの準備
                    X = df[config['features']].values
                    y = df[config['prediction_target']].values
                    
                    # スケーラーの読み込みと適用
                    feature_scaler = joblib.load(f"{self.models_path}/{model_name}_feature_scaler.joblib")
                    target_scaler = joblib.load(f"{self.models_path}/{model_name}_target_scaler.joblib")
                    
                    X = feature_scaler.transform(X)
                    
                    # LSTMの場合、シーケンスデータを準備
                    if config['model_type'] == 'lstm':
                        X = self.prepare_sequence_data(X)
                        y = y[10:]  # シーケンス長分のターゲットをスキップ
                    
                    # 予測
                    y_pred = model.predict(X)
                    
                    # スケーリング逆変換
                    y = target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
                    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                    
                    # 評価指標の計算
                    metrics = {
                        'prediction_vs_actual': list(zip(y.tolist(), y_pred.tolist())),
                        'residuals': (y - y_pred).tolist(),
                        'feature_importance': self.get_feature_importance(model, config['features'])
                    }
                    
                    regression_metrics[model_name] = metrics
                except Exception as e:
                    print(f"モデル {model_name} の評価中にエラーが発生しました: {str(e)}")
                    continue
        
        return regression_metrics
    
    def get_feature_importance(self, model, features: List[str]) -> Dict[str, float]:
        """特徴量の重要度を取得"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(features, importance.tolist()))
        return {}
    
    def find_best_models(self) -> Dict[str, str]:
        """各予測タスクで最も性能の良いモデルを特定"""
        best_models = {}
        
        # 分類タスク
        for target in [
            PredictionTargets.PRICE_DIRECTION,
            PredictionTargets.TREND_STRENGTH,
            PredictionTargets.TRADING_SIGNAL,
            PredictionTargets.VOLATILITY_LEVEL
        ]:
            best_model = None
            best_score = 0
            
            for model_name, result in self.results.items():
                if target in model_name and result['metrics'].get('accuracy'):
                    if result['metrics']['accuracy'] > best_score:
                        best_score = result['metrics']['accuracy']
                        best_model = model_name
            
            if best_model:
                best_models[target] = best_model
        
        # 回帰タスク
        for target in [
            PredictionTargets.PRICE_CHANGE,
            PredictionTargets.NEXT_PRICE,
            PredictionTargets.OPTIMAL_POSITION,
            PredictionTargets.RISK_REWARD
        ]:
            best_model = None
            best_score = float('inf')
            
            for model_name, result in self.results.items():
                if target in model_name and result['metrics'].get('mse'):
                    if result['metrics']['mse'] < best_score:
                        best_score = result['metrics']['mse']
                        best_model = model_name
            
            if best_model:
                best_models[target] = best_model
        
        return best_models
    
    def plot_evaluation_results(self, save_path: str = './results/evaluation_plots'):
        """評価結果のプロット"""
        os.makedirs(save_path, exist_ok=True)
        
        # 分類モデルの性能比較
        classification_scores = {
            name: result['metrics'].get('accuracy', 0)
            for name, result in self.results.items()
            if result['metrics'].get('accuracy') is not None
        }
        
        if classification_scores:
            plt.figure(figsize=(12, 6))
            plt.bar(classification_scores.keys(), classification_scores.values())
            plt.xticks(rotation=45)
            plt.title('Classification Models Accuracy Comparison')
            plt.tight_layout()
            plt.savefig(f"{save_path}/classification_comparison.png")
            plt.close()
        
        # 回帰モデルの性能比較
        regression_scores = {
            name: result['metrics'].get('rmse', 0)
            for name, result in self.results.items()
            if result['metrics'].get('rmse') is not None
        }
        
        if regression_scores:
            plt.figure(figsize=(12, 6))
            plt.bar(regression_scores.keys(), regression_scores.values())
            plt.xticks(rotation=45)
            plt.title('Regression Models RMSE Comparison')
            plt.tight_layout()
            plt.savefig(f"{save_path}/regression_comparison.png")
            plt.close()
    
    def generate_evaluation_report(self, df: pd.DataFrame) -> Dict:
        """評価レポートの生成"""
        # 分類モデルの評価
        classification_metrics = self.evaluate_classification_models(df)
        
        # 回帰モデルの評価
        regression_metrics = self.evaluate_regression_models(df)
        
        # 最適なモデルの特定
        best_models = self.find_best_models()
        
        # プロットの生成
        self.plot_evaluation_results()
        
        # レポートの作成
        report = {
            'best_models': best_models,
            'classification_metrics': classification_metrics,
            'regression_metrics': regression_metrics,
            'model_comparison': {
                name: result['metrics']
                for name, result in self.results.items()
            }
        }
        
        return report

def main():
    # データの読み込み
    df = pd.read_csv('./data/features_20250217.csv')
    
    # データの前処理
    df = preprocess_data(df)
    
    # モデル評価
    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(df)
    
    # 結果の保存
    with open('./results/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # 最適なモデルの表示
    print("\n最適なモデル:")
    for target, model in report['best_models'].items():
        print(f"{target}: {model}")
        print(f"メトリクス: {report['model_comparison'][model]}")

if __name__ == "__main__":
    main() 