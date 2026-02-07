#!/usr/bin/env python3
"""
SageMaker学習ジョブのエントリーポイント

Usage:
    python train.py --epochs 5 --batch-size 256 --lr 0.001
"""
import argparse
import os
import json
import sys

import torch

# ローカルモジュールのインポート
from model import FashionCNN, get_model
from dataset import load_data_simple
from trainer import Trainer, TrainingConfig


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description='Fashion-MNIST Training')
    
    # ハイパーパラメータ
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--model', type=str, default='default',
                        choices=['default', 'large'],
                        help='Model type (default: default)')
    
    # SageMaker環境変数
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--test', type=str,
                        default=os.environ.get('SM_CHANNEL_TESTING', '/opt/ml/input/data/testing'))
    
    return parser.parse_args()


def main():
    """メイン関数"""
    args = parse_args()
    
    print("=" * 60)
    print("Fashion-MNIST Training Job")
    print("=" * 60)
    print(f"Arguments: {vars(args)}")
    print()
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # データ読み込み
    print("📂 Loading data...")
    train_loader, test_loader = load_data_simple(
        train_dir=args.train,
        test_dir=args.test,
        batch_size=args.batch_size
    )
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    print()
    
    # モデル作成
    print("🏗️  Building model...")
    model = get_model(args.model)
    print(f"   Model: {args.model}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 学習設定
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Trainer作成
    trainer = Trainer(model, config, device)
    
    # 学習実行
    print("🚀 Starting training...")
    print("-" * 60)
    result = trainer.train(train_loader, test_loader)
    
    # モデル保存
    model_path = os.path.join(args.model_dir, 'model.pth')
    trainer.save_model(model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    # 結果を保存
    results_path = os.path.join(args.model_dir, 'results.json')
    result.save(results_path)
    print(f"📊 Results saved: {results_path}")
    
    # 最終結果を表示
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"📈 Final accuracy: {result.final_accuracy:.2%}")
    print(f"📈 Best accuracy:  {result.best_accuracy:.2%}")
    print(f"⏱️  Total time:     {result.total_time:.1f}s")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
