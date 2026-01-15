"""
Fine-tuning Demo After L1 Pruning

This script demonstrates how to fine-tune a pruned SegFormer model.
It provides a template for continuing training after applying L1 pruning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from nets.segformer import SegFormer
from nets.pruning import L1Pruner
import argparse


def create_dummy_dataloader(batch_size=2, num_batches=10, input_size=512, num_classes=21):
    """
    Create a dummy dataloader for demonstration purposes.
    Replace this with your actual data loading logic.
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, input_size, num_classes):
            self.num_samples = num_samples
            self.input_size = input_size
            self.num_classes = num_classes
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate random image and label
            image = torch.randn(3, self.input_size, self.input_size)
            label = torch.randint(0, self.num_classes, (self.input_size, self.input_size))
            return image, label
    
    dataset = DummyDataset(num_batches * batch_size, input_size, num_classes)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return dataloader


def fine_tune_pruned_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=1e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Fine-tune a pruned model.
    
    Args:
        model: Pruned model to fine-tune
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: Device to train on
    """
    
    print(f"\n{'='*60}")
    print("  Fine-tuning Pruned Model")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Move model to device
    model = model.to(device)
    model.train()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Handle case where model returns tuple (logits, car_out)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            if (batch_idx + 1) % 5 == 0:
                avg_loss = train_loss / train_samples
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}")
        
        # Calculate epoch statistics
        epoch_loss = train_loss / train_samples
        print(f"\nTraining Loss: {epoch_loss:.4f}")
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    val_samples += images.size(0)
            
            val_loss = val_loss / val_samples
            print(f"Validation Loss: {val_loss:.4f}")
    
    print(f"\n{'='*60}")
    print("  Fine-tuning Complete!")
    print(f"{'='*60}\n")
    
    return model


def main():
    """Main function demonstrating the complete pruning and fine-tuning workflow."""
    
    parser = argparse.ArgumentParser(
        description='Fine-tuning Demo After L1 Pruning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num_classes', type=int, default=21,
                        help='Number of output classes')
    parser.add_argument('--phi', type=str, default='b0',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        help='SegFormer model variant')
    parser.add_argument('--pruning_ratio', type=float, default=0.3,
                        help='Ratio of channels to prune (0.0-1.0)')
    parser.add_argument('--input_size', type=int, default=128,
                        help='Input image size (square)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--save_path', type=str, default='finetuned_pruned_model.pth',
                        help='Path to save fine-tuned model')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"{'='*60}")
    print("  L1 Pruning + Fine-tuning Workflow")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Model: SegFormer-{args.phi}")
    print(f"  Number of classes: {args.num_classes}")
    print(f"  Pruning ratio: {args.pruning_ratio}")
    print(f"  Input size: {args.input_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {device}")
    
    # Step 1: Create original model
    print(f"\n{'='*60}")
    print("  Step 1: Creating Original Model")
    print(f"{'='*60}")
    model = SegFormer(num_classes=args.num_classes, phi=args.phi, pretrained=False)
    print(f"✓ Model created")
    
    # Step 2: Apply L1 Pruning
    print(f"\n{'='*60}")
    print("  Step 2: Applying L1 Pruning")
    print(f"{'='*60}")
    pruner = L1Pruner(model, pruning_ratio=args.pruning_ratio)
    pruned_model = pruner.prune()
    
    stats = pruner.get_pruning_statistics()
    print(f"\n✓ Pruning complete:")
    print(f"  - Pruned channels: {stats['pruned_channels']}/{stats['total_channels']}")
    print(f"  - Pruning ratio: {stats['pruning_ratio_actual']:.2%}")
    
    # Step 3: Create data loaders (dummy data for demo)
    print(f"\n{'='*60}")
    print("  Step 3: Creating Data Loaders")
    print(f"{'='*60}")
    print("  Note: Using dummy data for demonstration")
    print("  Replace with your actual data loading logic")
    
    train_loader = create_dummy_dataloader(
        batch_size=args.batch_size,
        num_batches=10,
        input_size=args.input_size,
        num_classes=args.num_classes
    )
    
    val_loader = create_dummy_dataloader(
        batch_size=args.batch_size,
        num_batches=3,
        input_size=args.input_size,
        num_classes=args.num_classes
    )
    
    print(f"✓ Data loaders created")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    
    # Step 4: Fine-tune pruned model
    print(f"\n{'='*60}")
    print("  Step 4: Fine-tuning Pruned Model")
    print(f"{'='*60}")
    
    finetuned_model = fine_tune_pruned_model(
        pruned_model,
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Step 5: Save fine-tuned model
    print(f"{'='*60}")
    print("  Step 5: Saving Fine-tuned Model")
    print(f"{'='*60}")
    torch.save(finetuned_model.state_dict(), args.save_path)
    print(f"✓ Model saved to: {args.save_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("  Workflow Complete!")
    print(f"{'='*60}")
    print("✓ Model pruned and fine-tuned successfully")
    print(f"✓ Final model saved to: {args.save_path}")
    print("\nNext steps:")
    print("  1. Evaluate the fine-tuned model on your test set")
    print("  2. Compare accuracy with original model")
    print("  3. Deploy the compressed model")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
