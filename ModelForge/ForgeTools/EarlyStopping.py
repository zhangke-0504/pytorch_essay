import os
import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, save_path='./models', verbose=True):
        """
        升级版早停类，支持双指标监控[5](@ref)
        
        参数：
            patience: 容忍无改进的epoch数
            delta: 最小改进阈值
            save_path: 模型保存路径
            verbose: 是否打印提示信息
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.verbose = verbose
        
        self.best_val = float('inf')
        self.best_train = float('inf')
        self.counter = 0
        self.best_model = None
        os.makedirs(save_path, exist_ok=True)

    def __call__(self, train_loss, val_loss, model, epoch):
        """返回是否触发早停"""
        improved = (val_loss < self.best_val - self.delta)
        # 早停布尔值
        should_stop = False
        # 学习率更新布尔值
        should_scheduler_step = False
               
        if improved:  # 验证损失有显著改进
            self.best_val = val_loss
            self.best_train = train_loss
            self.counter = 0
            self._save_model(model, epoch)
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            step_count_list = []
            for i in range(5):
                step_count_list.append(int(self.patience * 0.25) + i + 1)
            if self.counter in step_count_list:
                # 如果早停计数器超过patience一半，则需要更新学习率
                should_scheduler_step = True

        # 触发早停条件
        if self.counter >= self.patience:
            if self.verbose:
                print(f'Early stopping triggered at epoch {epoch}')
            should_stop = True
            return should_stop, should_scheduler_step
        return should_stop, should_scheduler_step

    def _save_model(self, model, epoch):
        """保存最佳模型"""
        self.best_model = model.state_dict()
        torch.save(self.best_model, 
                 os.path.join(self.save_path, 'best.pt'))
        if self.verbose:
            print(f"Saved best model with val_loss: {self.best_val:.4f} at epoch {epoch}")