#!/bin/bash
# =============================================================================
# MLOps Fashion-MNIST Demo - クリーンアップスクリプト
#
# 作成したすべてのAWSリソースを削除します
# =============================================================================

set -e

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() { echo -e "${GREEN}[STEP]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 設定
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region 2>/dev/null || echo "ap-northeast-1")
STACK_NAME="mlops-demo"
BUCKET_NAME="mlops-demo-${ACCOUNT_ID}"

echo ""
echo "=============================================="
echo "  MLOps Demo リソース削除"
echo "=============================================="
echo ""
echo "以下のリソースが削除されます:"
echo "  - CloudFormation Stack: ${STACK_NAME}"
echo "  - S3 Bucket: ${BUCKET_NAME}"
echo "  - 関連するすべてのリソース (Lambda, Step Functions, IAM Roles, etc.)"
echo ""

read -p "本当に削除しますか？ (yes を入力): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "キャンセルしました"
    exit 0
fi

echo ""

# -----------------------------------------------------------------------------
# SageMakerエンドポイントの削除（存在する場合）
# -----------------------------------------------------------------------------
print_step "SageMakerリソースを確認中..."

# エンドポイントの削除
ENDPOINTS=$(aws sagemaker list-endpoints \
    --name-contains "fashion" \
    --query 'Endpoints[].EndpointName' \
    --output text --region ${REGION} 2>/dev/null || echo "")

for endpoint in $ENDPOINTS; do
    print_warn "エンドポイントを削除: ${endpoint}"
    aws sagemaker delete-endpoint --endpoint-name ${endpoint} --region ${REGION} 2>/dev/null || true
done

# エンドポイント設定の削除
ENDPOINT_CONFIGS=$(aws sagemaker list-endpoint-configs \
    --name-contains "fashion" \
    --query 'EndpointConfigs[].EndpointConfigName' \
    --output text --region ${REGION} 2>/dev/null || echo "")

for config in $ENDPOINT_CONFIGS; do
    print_warn "エンドポイント設定を削除: ${config}"
    aws sagemaker delete-endpoint-config --endpoint-config-name ${config} --region ${REGION} 2>/dev/null || true
done

# モデルの削除
MODELS=$(aws sagemaker list-models \
    --name-contains "model-fashion" \
    --query 'Models[].ModelName' \
    --output text --region ${REGION} 2>/dev/null || echo "")

for model in $MODELS; do
    print_warn "モデルを削除: ${model}"
    aws sagemaker delete-model --model-name ${model} --region ${REGION} 2>/dev/null || true
done

# -----------------------------------------------------------------------------
# S3バケットの空化と削除
# -----------------------------------------------------------------------------
print_step "S3バケットを空にしています..."

if aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null; then
    aws s3 rm "s3://${BUCKET_NAME}" --recursive
    print_warn "バケットの中身を削除しました"
else
    print_warn "バケットが見つかりません（既に削除済み）"
fi

# -----------------------------------------------------------------------------
# CloudFormationスタックの削除
# -----------------------------------------------------------------------------
print_step "CloudFormationスタックを削除中..."

if aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${REGION} 2>/dev/null; then
    aws cloudformation delete-stack --stack-name ${STACK_NAME} --region ${REGION}
    
    echo "スタック削除を待機中..."
    aws cloudformation wait stack-delete-complete --stack-name ${STACK_NAME} --region ${REGION}
    print_warn "CloudFormationスタックを削除しました"
else
    print_warn "スタックが見つかりません（既に削除済み）"
fi

# -----------------------------------------------------------------------------
# S3バケットの削除（CloudFormationで作成されていない場合）
# -----------------------------------------------------------------------------
print_step "S3バケットを削除中..."

if aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null; then
    aws s3 rb "s3://${BUCKET_NAME}" --force
    print_warn "S3バケットを削除しました"
else
    print_warn "バケットは既に削除されています"
fi

# -----------------------------------------------------------------------------
# 完了
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}✅ クリーンアップ完了${NC}"
echo "=============================================="
echo ""
echo "すべてのリソースが削除されました。"
echo ""
