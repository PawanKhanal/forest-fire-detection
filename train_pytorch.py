"""Training script for forest fire detection using PyTorch."""

import argparse
import torch
from pathlib import Path
import sys
from PIL import Image

# Allow large images (disable decompression bomb warning)
Image.MAX_IMAGE_PIXELS = None

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.cnn_model_pytorch import FireDetectionTrainer, ForestFireCNN
from src.data.pytorch_dataset import ForestFireDataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Forest Fire Detection Model')
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='data/raw',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=15,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001,
        help='Learning rate'
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    print("=" * 60)
    
    # Create data loaders
    print("\n📦 Loading dataset...")
    data_loader = ForestFireDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    train_loader, val_loader, test_loader = data_loader.create_loaders()
    class_weights = data_loader.get_class_weights()
    class_names = data_loader.get_class_names()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Classes: {class_names}")
    print(f"   Class weights: {class_weights.tolist()}")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n🏗️  Building model...")
    model = ForestFireCNN(num_classes=len(class_names))
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = FireDetectionTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        class_weights=class_weights
    )
    
    # Train model
    print(f"\n🔥 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs
    )
    
    # Final evaluation on test set
    print("\n📊 Final Test Evaluation")
    print("=" * 60)
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save final model
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'forest_fire_cnn_final.pth'
    model.save(str(save_path))
    print(f"\n✅ Model saved to {save_path}")
    
    # Print training summary
    print("\n📈 Training Summary:")
    print("=" * 60)
    print(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()