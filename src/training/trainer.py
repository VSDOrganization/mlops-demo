"""
学習ロジックモジュール
"""
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """学習設定"""
    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler_step_size: int = 3
    scheduler_gamma: float = 0.1


@dataclass
class EpochResult:
    """1エポックの結果"""
    epoch: int
    train_loss: float
    train_accuracy: float
    test_accuracy: float
    epoch_time: float


@dataclass
class TrainingResult:
    """学習全体の結果"""
    final_accuracy: float
    best_accuracy: float
    total_time: float
    epochs_completed: int
    history: List[dict]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Trainer:
    """
    モデル学習を管理するクラス
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: 学習するモデル
            config: 学習設定
            device: 使用デバイス（Noneの場合は自動検出）
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # 損失関数
        self.criterion = nn.CrossEntropyLoss()
        
        # オプティマイザ
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学習率スケジューラ
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
        
        # 履歴
        self.history: List[EpochResult] = []
    
    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        1エポック分の学習を実行
        
        Args:
            train_loader: 学習データローダー
            
        Returns:
            (平均損失, 正解率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 順伝播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            # 統計
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> float:
        """
        モデルを評価
        
        Args:
            test_loader: テストデータローダー
            
        Returns:
            正解率
        """
        self.model.eval()
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return correct / total
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True
    ) -> TrainingResult:
        """
        学習を実行
        
        Args:
            train_loader: 学習データローダー
            test_loader: テストデータローダー
            verbose: 進捗表示
            
        Returns:
            学習結果
        """
        if verbose:
            print(f"🖥️  Device: {self.device}")
            print(f"📊 Training samples: {len(train_loader.dataset)}")
            print(f"📊 Test samples: {len(test_loader.dataset)}")
            print(f"⚙️  Epochs: {self.config.epochs}")
            print(f"⚙️  Learning rate: {self.config.learning_rate}")
            print("-" * 60)
        
        start_time = time.time()
        best_accuracy = 0.0
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # 学習
            train_loss, train_acc = self.train_one_epoch(train_loader)
            
            # 評価
            test_acc = self.evaluate(test_loader)
            
            # スケジューラ更新
            self.scheduler.step()
            
            # 結果を記録
            epoch_time = time.time() - epoch_start
            result = EpochResult(
                epoch=epoch,
                train_loss=round(train_loss, 4),
                train_accuracy=round(train_acc, 4),
                test_accuracy=round(test_acc, 4),
                epoch_time=round(epoch_time, 2)
            )
            self.history.append(result)
            
            # ベスト更新
            if test_acc > best_accuracy:
                best_accuracy = test_acc
            
            if verbose:
                print(
                    f"Epoch {epoch:2d}/{self.config.epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Train: {train_acc:.2%} | "
                    f"Test: {test_acc:.2%} | "
                    f"Time: {epoch_time:.1f}s"
                )
        
        total_time = time.time() - start_time
        
        if verbose:
            print("-" * 60)
            print(f"✅ Training completed in {total_time:.1f}s")
            print(f"📈 Best accuracy: {best_accuracy:.2%}")
        
        return TrainingResult(
            final_accuracy=round(self.history[-1].test_accuracy, 4),
            best_accuracy=round(best_accuracy, 4),
            total_time=round(total_time, 2),
            epochs_completed=self.config.epochs,
            history=[asdict(h) for h in self.history]
        )
    
    def save_model(self, path: str):
        """モデルを保存"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """モデルを読み込み"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def create_trainer(
    model: nn.Module,
    epochs: int = 5,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None
) -> Trainer:
    """
    Trainerを簡単に作成するヘルパー関数
    """
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate
    )
    return Trainer(model, config, device)
