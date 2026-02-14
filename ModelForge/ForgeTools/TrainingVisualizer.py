import matplotlib.pyplot as plt
import os

class TrainingVisualizer:
    def __init__(self, save_dir="./visualization"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        # 初始化数据容器
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epochs = []

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """更新训练数据"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        
        # 实时更新图像
        self._plot_loss_curve()
        self._plot_accuracy_curve()

    def _plot_loss_curve(self):
        """绘制损失曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Train Loss')
        plt.plot(self.epochs, self.val_losses, 'r-', label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'))
        plt.close()

    def _plot_accuracy_curve(self):
        """绘制准确率曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.epochs, self.train_accuracies, 'g--', label='Train Accuracy')
        plt.plot(self.epochs, self.val_accuracies, 'm--', label='Val Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'accuracy_curve.png'))
        plt.close()