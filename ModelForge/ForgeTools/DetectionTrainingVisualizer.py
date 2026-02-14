import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def plot_acc_loss_curve(history):
    """Plot accuracy curves during training"""
    epochs = list(range(1, len(history['train_acc']) + 1))
    
    plt.figure(figsize=(15, 6))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'go-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'mo-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.1])  # Ensure space above 1.0 for labels
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save image
    save_path = os.path.join(ROOT_DIR, 'visualization' , 'Detection_acc_loss_curve.png')
    plt.savefig(save_path, dpi=300)
    print(f"Accuracy curves saved to: {save_path}")
    plt.close()