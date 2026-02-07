#!/usr/bin/env python3
"""
Fashion-MNISTデータセットをS3に準備するスクリプト

Usage:
    python prepare_dataset.py --bucket my-bucket --region ap-northeast-1
"""
import argparse
import gzip
import io
import json
import urllib.request
from datetime import datetime
from typing import Tuple

import boto3
import numpy as np


# Fashion-MNISTのダウンロードURL
FASHION_MNIST_BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# クラスラベル
FASHION_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def download_and_parse_images(url: str) -> np.ndarray:
    """
    画像データをダウンロードしてNumPy配列に変換
    
    Args:
        url: ダウンロードURL
        
    Returns:
        画像配列 (N, 28, 28)
    """
    print(f"   Downloading: {url.split('/')[-1]}")
    with urllib.request.urlopen(url) as response:
        compressed_data = response.read()
    
    with gzip.open(io.BytesIO(compressed_data), 'rb') as f:
        # IDXファイルフォーマットのヘッダーをスキップ（16バイト）
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    return data.reshape(-1, 28, 28)


def download_and_parse_labels(url: str) -> np.ndarray:
    """
    ラベルデータをダウンロードしてNumPy配列に変換
    
    Args:
        url: ダウンロードURL
        
    Returns:
        ラベル配列 (N,)
    """
    print(f"   Downloading: {url.split('/')[-1]}")
    with urllib.request.urlopen(url) as response:
        compressed_data = response.read()
    
    with gzip.open(io.BytesIO(compressed_data), 'rb') as f:
        # IDXファイルフォーマットのヘッダーをスキップ（8バイト）
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    return data


def download_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fashion-MNISTデータセットをダウンロード
    
    Returns:
        (train_images, train_labels, test_images, test_labels)
    """
    print("📥 Downloading Fashion-MNIST dataset...")
    
    train_images = download_and_parse_images(
        FASHION_MNIST_BASE_URL + FASHION_MNIST_FILES["train_images"]
    )
    train_labels = download_and_parse_labels(
        FASHION_MNIST_BASE_URL + FASHION_MNIST_FILES["train_labels"]
    )
    test_images = download_and_parse_images(
        FASHION_MNIST_BASE_URL + FASHION_MNIST_FILES["test_images"]
    )
    test_labels = download_and_parse_labels(
        FASHION_MNIST_BASE_URL + FASHION_MNIST_FILES["test_labels"]
    )
    
    print(f"   ✅ Train: {train_images.shape[0]} samples")
    print(f"   ✅ Test:  {test_images.shape[0]} samples")
    
    return train_images, train_labels, test_images, test_labels


def upload_to_s3(
    bucket: str,
    region: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray
) -> None:
    """
    データセットをS3にアップロード
    
    Args:
        bucket: S3バケット名
        region: AWSリージョン
        train_images, train_labels, test_images, test_labels: データ
    """
    print(f"\n📤 Uploading to S3: s3://{bucket}/")
    
    s3_client = boto3.client('s3', region_name=region)
    
    # 学習データ
    print("   Uploading training data...")
    train_buffer = io.BytesIO()
    np.savez_compressed(train_buffer, images=train_images, labels=train_labels)
    train_buffer.seek(0)
    s3_client.upload_fileobj(train_buffer, bucket, 'training/train.npz')
    print("   ✅ training/train.npz")
    
    # テストデータ
    print("   Uploading test data...")
    test_buffer = io.BytesIO()
    np.savez_compressed(test_buffer, images=test_images, labels=test_labels)
    test_buffer.seek(0)
    s3_client.upload_fileobj(test_buffer, bucket, 'testing/test.npz')
    print("   ✅ testing/test.npz")
    
    # メタデータ
    metadata = {
        "dataset": "Fashion-MNIST",
        "created_at": datetime.now().isoformat(),
        "train_samples": int(train_images.shape[0]),
        "test_samples": int(test_images.shape[0]),
        "image_shape": [28, 28],
        "num_classes": 10,
        "labels": FASHION_LABELS
    }
    s3_client.put_object(
        Bucket=bucket,
        Key='metadata.json',
        Body=json.dumps(metadata, indent=2),
        ContentType='application/json'
    )
    print("   ✅ metadata.json")


def create_bucket_if_not_exists(bucket: str, region: str) -> None:
    """
    S3バケットが存在しない場合は作成
    
    Args:
        bucket: バケット名
        region: リージョン
    """
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"ℹ️  Bucket already exists: {bucket}")
    except:
        print(f"📦 Creating bucket: {bucket}")
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket)
        else:
            s3_client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"   ✅ Bucket created")


def main():
    parser = argparse.ArgumentParser(description='Prepare Fashion-MNIST dataset')
    parser.add_argument('--bucket', type=str, required=True,
                        help='S3 bucket name')
    parser.add_argument('--region', type=str, default='ap-northeast-1',
                        help='AWS region (default: ap-northeast-1)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fashion-MNIST Dataset Preparation")
    print("=" * 60)
    print(f"Bucket: {args.bucket}")
    print(f"Region: {args.region}")
    print()
    
    # バケット作成
    create_bucket_if_not_exists(args.bucket, args.region)
    
    # データダウンロード
    train_x, train_y, test_x, test_y = download_fashion_mnist()
    
    # S3にアップロード
    upload_to_s3(
        args.bucket, args.region,
        train_x, train_y, test_x, test_y
    )
    
    print()
    print("=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
