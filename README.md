# MLOps Fashion-MNIST Demo

AWS SageMakerを使用したMLOpsパイプラインのデモプロジェクトです。

## 概要

- **データセット**: Fashion-MNIST（10クラスのファッションアイテム画像）
- **モデル**: シンプルなCNN（PyTorch）
- **インフラ**: AWS SageMaker + Step Functions + EventBridge
- **自動実行**: 毎日0:00 JSTに学習→評価→デプロイ

## アーキテクチャ

```
EventBridge (毎日0:00 JST)
    ↓
Step Functions (ワークフロー制御)
    ↓
SageMaker Training (GPU学習: ml.g4dn.xlarge)
    ↓
Lambda (精度評価: 85%以上?)
    ↓
SageMaker Endpoint (デプロイ) or 通知 (精度未達)
```

## ディレクトリ構成

```
mlops-fashion-mnist-demo/
├── README.md
├── LICENSE
├── setup.sh                    # ワンクリックセットアップ
├── cleanup.sh                  # リソース削除
├── .gitignore
│
├── src/
│   ├── data_preparation/
│   │   └── prepare_dataset.py  # データセット準備
│   ├── training/
│   │   ├── train.py            # 学習エントリーポイント
│   │   ├── model.py            # モデル定義
│   │   ├── trainer.py          # 学習ロジック
│   │   ├── dataset.py          # データローダー
│   │   └── requirements.txt
│   └── lambda/
│       └── accuracy_checker.py # 精度評価Lambda
│
└── infrastructure/
    └── cloudformation/
        └── mlops-stack.yaml    # AWSリソース定義
```

## クイックスタート

### 前提条件

- AWSアカウント
- AWS CLI設定済み（`aws configure`）

### セットアップ（AWS CloudShellで実行）

```bash
# 1. リポジトリをクローン
git clone https://github.com/your-username/mlops-fashion-mnist-demo.git
cd mlops-fashion-mnist-demo

# 2. セットアップ実行（約5分）
chmod +x setup.sh
./setup.sh

# 3. 手動テスト実行
aws stepfunctions start-execution \
  --state-machine-arn $(aws cloudformation describe-stacks \
    --stack-name mlops-demo \
    --query 'Stacks[0].Outputs[?OutputKey==`PipelineArn`].OutputValue' \
    --output text)
```

### 実行状況の確認

AWS Step Functionsコンソールで確認:
https://console.aws.amazon.com/states/

## コスト目安

| 項目 | 料金 |
|------|------|
| SageMaker学習 (ml.g4dn.xlarge) | 約$0.75/回 |
| 月額（毎日実行） | 約$23/月 |

※学習完了後は自動停止するため、使用時間分のみ課金

## 環境削除

```bash
./cleanup.sh
```

## カスタマイズ

### 精度閾値の変更

`infrastructure/cloudformation/mlops-stack.yaml` の `THRESHOLD` を変更:

```yaml
Environment:
  Variables:
    THRESHOLD: "0.90"  # 90%に変更
```

### エポック数の変更

同ファイルの `HyperParameters` を変更:

```yaml
HyperParameters:
  epochs: "\"10\""  # 10エポックに変更
```

## ライセンス

MIT License
