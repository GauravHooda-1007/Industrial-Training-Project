# ======== Imports and Config ========
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

data_dir = "C:/Users/gaura/Desktop/OAS/OAS_png_dataset"
num_epochs = 5
batch_size = 32
learning_rate = 1e-4
image_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)
if device.type == 'cuda':
    print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not available, training on CPU.")


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ======== Main Entry Point ========
if __name__ == "__main__":
    # Data loading
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss with class weights
    class_counts = [len(os.listdir(os.path.join(data_dir, c))) for c in class_names]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc="🔁 Training", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        print(f"✅ Train Loss: {running_loss/len(train_loader.dataset):.4f}, Accuracy: {correct/total:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="📊 Validating", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()

        print(f"📈 Val Loss: {val_loss/len(val_loader.dataset):.4f}, Accuracy: {val_correct/val_total:.4f}")

    torch.save(model.state_dict(), "C:\Users\gaura\Desktop\OAS\alzheimer_resnet18.pth")
    print("💾 Model saved as alzheimer_resnet18.pth")
