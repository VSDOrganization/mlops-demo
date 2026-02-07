"""
Fashion-MNIST用CNNモデル定義
"""
import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """
    Fashion-MNIST分類用のシンプルなCNNモデル
    
    アーキテクチャ:
        - 3層の畳み込みブロック（Conv2d + BatchNorm + ReLU + Pooling）
        - 2層の全結合層（Dropout付き）
    
    入力: (batch_size, 1, 28, 28) - グレースケール画像
    出力: (batch_size, 10) - 10クラスのロジット
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        """
        Args:
            num_classes: 出力クラス数（デフォルト: 10）
            dropout_rate: ドロップアウト率（デフォルト: 0.3）
        """
        super().__init__()
        
        # 特徴抽出部分
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 7x7 -> 1x1
        )
        
        # 分類部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル (batch_size, 1, 28, 28)
            
        Returns:
            出力ロジット (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_num_parameters(self) -> int:
        """学習可能なパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FashionCNNLarge(nn.Module):
    """
    より大きなCNNモデル（高精度版）
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.4):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name: str = "default", **kwargs) -> nn.Module:
    """
    モデルファクトリー関数
    
    Args:
        model_name: モデル名 ("default" or "large")
        **kwargs: モデルに渡す追加引数
        
    Returns:
        モデルインスタンス
    """
    models = {
        "default": FashionCNN,
        "large": FashionCNNLarge,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


# クラスラベル定義
FASHION_LABELS = [
    "T-shirt/top",  # Tシャツ
    "Trouser",      # ズボン
    "Pullover",     # セーター
    "Dress",        # ドレス
    "Coat",         # コート
    "Sandal",       # サンダル
    "Shirt",        # シャツ
    "Sneaker",      # スニーカー
    "Bag",          # バッグ
    "Ankle boot",   # ブーツ
]

FASHION_LABELS_JA = [
    "Tシャツ",
    "ズボン",
    "セーター",
    "ドレス",
    "コート",
    "サンダル",
    "シャツ",
    "スニーカー",
    "バッグ",
    "ブーツ",
]
