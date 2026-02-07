"""
データセット読み込みモジュール
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Optional


class FashionMNISTDataset(Dataset):
    """
    Fashion-MNISTデータセットクラス
    
    NPZファイルからデータを読み込み、PyTorchのDatasetとして提供
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[callable] = None,
        normalize: bool = True
    ):
        """
        Args:
            data_path: NPZファイルのパス
            transform: データ変換関数（オプション）
            normalize: 0-1に正規化するか
        """
        self.transform = transform
        
        # NPZファイルを読み込み
        data = np.load(data_path)
        self.images = data['images'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        
        # 正規化 (0-255 -> 0-1)
        if normalize:
            self.images = self.images / 255.0
        
        # チャネル次元を追加 (N, H, W) -> (N, 1, H, W)
        self.images = self.images[:, np.newaxis, :, :]
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_data_from_npz(
    train_path: str,
    test_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    NPZファイルからDataLoaderを作成
    
    Args:
        train_path: 学習データのNPZファイルパス
        test_path: テストデータのNPZファイルパス
        batch_size: バッチサイズ
        num_workers: データローダーのワーカー数
        pin_memory: GPUメモリへのピン留め
        
    Returns:
        (train_loader, test_loader)
    """
    train_dataset = FashionMNISTDataset(train_path)
    test_dataset = FashionMNISTDataset(test_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def load_data_simple(
    train_dir: str,
    test_dir: str,
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    シンプルなデータ読み込み（SageMaker用）
    
    Args:
        train_dir: 学習データディレクトリ
        test_dir: テストデータディレクトリ
        batch_size: バッチサイズ
        
    Returns:
        (train_loader, test_loader)
    """
    # 学習データ
    train_npz = os.path.join(train_dir, 'train.npz')
    train_data = np.load(train_npz)
    train_images = train_data['images'].astype(np.float32) / 255.0
    train_labels = train_data['labels'].astype(np.int64)
    train_images = train_images[:, np.newaxis, :, :]
    
    train_x = torch.from_numpy(train_images)
    train_y = torch.from_numpy(train_labels)
    train_dataset = TensorDataset(train_x, train_y)
    
    # テストデータ
    test_npz = os.path.join(test_dir, 'test.npz')
    test_data = np.load(test_npz)
    test_images = test_data['images'].astype(np.float32) / 255.0
    test_labels = test_data['labels'].astype(np.int64)
    test_images = test_images[:, np.newaxis, :, :]
    
    test_x = torch.from_numpy(test_images)
    test_y = torch.from_numpy(test_labels)
    test_dataset = TensorDataset(test_x, test_y)
    
    # DataLoader作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_data_stats(data_loader: DataLoader) -> dict:
    """
    データセットの統計情報を取得
    
    Args:
        data_loader: DataLoader
        
    Returns:
        統計情報の辞書
    """
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy().tolist())
    
    unique, counts = np.unique(all_labels, return_counts=True)
    
    return {
        'total_samples': len(all_labels),
        'num_classes': len(unique),
        'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
        'batch_size': data_loader.batch_size,
        'num_batches': len(data_loader)
    }
