#!/bin/bash
# =============================================================================
# MLOps Fashion-MNIST Demo - セットアップスクリプト
# 
# 使用方法:
#   chmod +x setup.sh
#   ./setup.sh
#
# 前提条件:
#   - AWS CLI設定済み (aws configure)
#   - Python 3.9以上
# =============================================================================

set -e

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() { echo -e "${GREEN}[STEP]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=============================================="
echo "  MLOps Fashion-MNIST Demo セットアップ"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# 前提条件チェック
# -----------------------------------------------------------------------------
print_step "前提条件を確認中..."

# AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLIがインストールされていません"
    echo "インストール: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    exit 1
fi
print_info "AWS CLI: $(aws --version | cut -d' ' -f1)"

# AWS認証情報
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS認証情報が設定されていません"
    echo "実行: aws configure"
    exit 1
fi

# Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3がインストールされていません"
    exit 1
fi
print_info "Python: $(python3 --version)"

# pip
if ! command -v pip3 &> /dev/null; then
    print_warn "pip3が見つかりません。python3 -m pipを使用します"
    PIP_CMD="python3 -m pip"
else
    PIP_CMD="pip3"
fi

# -----------------------------------------------------------------------------
# 設定値
# -----------------------------------------------------------------------------
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region 2>/dev/null || echo "ap-northeast-1")
STACK_NAME="mlops-demo"
BUCKET_NAME="mlops-demo-${ACCOUNT_ID}"

echo ""
print_info "AWS Account ID: ${ACCOUNT_ID}"
print_info "Region: ${REGION}"
print_info "Stack Name: ${STACK_NAME}"
print_info "S3 Bucket: ${BUCKET_NAME}"
echo ""

read -p "この設定で続行しますか？ (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "キャンセルしました"
    exit 0
fi

echo ""

# -----------------------------------------------------------------------------
# 依存パッケージのインストール
# -----------------------------------------------------------------------------
print_step "Python依存パッケージをインストール中..."
$PIP_CMD install boto3 numpy --quiet
print_info "インストール完了"

# -----------------------------------------------------------------------------
# S3バケットの作成
# -----------------------------------------------------------------------------
print_step "S3バケットを作成中..."

if aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null; then
    print_info "バケットは既に存在: ${BUCKET_NAME}"
else
    if [ "$REGION" = "us-east-1" ]; then
        aws s3 mb "s3://${BUCKET_NAME}" --region ${REGION}
    else
        aws s3 mb "s3://${BUCKET_NAME}" --region ${REGION}
    fi
    print_info "バケットを作成: ${BUCKET_NAME}"
fi

# -----------------------------------------------------------------------------
# データセットの準備
# -----------------------------------------------------------------------------
print_step "Fashion-MNISTデータセットを準備中..."
cd "${SCRIPT_DIR}/src/data_preparation"
python3 prepare_dataset.py --bucket ${BUCKET_NAME} --region ${REGION}
cd "${SCRIPT_DIR}"

# -----------------------------------------------------------------------------
# 学習コードのアップロード
# -----------------------------------------------------------------------------
print_step "学習コードをS3にアップロード中..."

# sourcedir.tar.gzを作成
cd "${SCRIPT_DIR}/src/training"
tar -czvf /tmp/sourcedir.tar.gz *.py requirements.txt 2>/dev/null
aws s3 cp /tmp/sourcedir.tar.gz "s3://${BUCKET_NAME}/code/sourcedir.tar.gz"
rm /tmp/sourcedir.tar.gz
cd "${SCRIPT_DIR}"
print_info "学習コードをアップロード: s3://${BUCKET_NAME}/code/sourcedir.tar.gz"

# -----------------------------------------------------------------------------
# Lambda関数のアップロード
# -----------------------------------------------------------------------------
print_step "Lambda関数をS3にアップロード中..."

cd "${SCRIPT_DIR}/src/lambda"
zip -j /tmp/accuracy_checker.zip accuracy_checker.py
aws s3 cp /tmp/accuracy_checker.zip "s3://${BUCKET_NAME}/lambda/accuracy_checker.zip"
rm /tmp/accuracy_checker.zip
cd "${SCRIPT_DIR}"
print_info "Lambda関数をアップロード: s3://${BUCKET_NAME}/lambda/accuracy_checker.zip"

# -----------------------------------------------------------------------------
# CloudFormationスタックのデプロイ
# -----------------------------------------------------------------------------
print_step "CloudFormationスタックをデプロイ中..."

aws cloudformation deploy \
    --template-file "${SCRIPT_DIR}/infrastructure/cloudformation/mlops-stack.yaml" \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
        BucketName=${BUCKET_NAME} \
        AccuracyThreshold=0.85 \
        TrainingInstanceType=ml.g4dn.xlarge \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION}

print_info "CloudFormationスタックのデプロイ完了"

# -----------------------------------------------------------------------------
# 出力情報の取得
# -----------------------------------------------------------------------------
print_step "リソース情報を取得中..."

PIPELINE_ARN=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --query 'Stacks[0].Outputs[?OutputKey==`PipelineArn`].OutputValue' \
    --output text --region ${REGION})

SNS_TOPIC_ARN=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --query 'Stacks[0].Outputs[?OutputKey==`SNSTopicArn`].OutputValue' \
    --output text --region ${REGION})

# -----------------------------------------------------------------------------
# 完了メッセージ
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}✅ セットアップ完了！${NC}"
echo "=============================================="
echo ""
echo "📌 作成されたリソース:"
echo "   - S3 Bucket: ${BUCKET_NAME}"
echo "   - Step Functions: ${PIPELINE_ARN}"
echo "   - SNS Topic: ${SNS_TOPIC_ARN}"
echo ""
echo "🚀 手動実行（テスト）:"
echo "   aws stepfunctions start-execution \\"
echo "     --state-machine-arn ${PIPELINE_ARN}"
echo ""
echo "📧 メール通知を受け取る場合:"
echo "   aws sns subscribe \\"
echo "     --topic-arn ${SNS_TOPIC_ARN} \\"
echo "     --protocol email \\"
echo "     --notification-endpoint your-email@example.com"
echo ""
echo "📅 自動実行:"
echo "   毎日 0:00 JST (UTC 15:00) に自動実行されます"
echo ""
echo "🔍 確認:"
echo "   https://${REGION}.console.aws.amazon.com/states/home?region=${REGION}#/statemachines/view/${PIPELINE_ARN}"
echo ""
echo "🧹 リソース削除:"
echo "   ./cleanup.sh"
echo ""
