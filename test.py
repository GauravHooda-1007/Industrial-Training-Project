import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Config
dataset_dir = "C:/Users/gaura/Desktop/OAS/OAS_test"
model_path = "C:/Users/gaura/Desktop/OAS/alzheimer_resnet18.pth"
class_names = ['AD', 'EMCI', 'LMCI', 'NC']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load model
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax().item()
        return pred_idx

# Evaluation setup
y_true = []
y_pred = []

print("\n🎯 Predicting 5 random images per class:\n")
for label_idx, label in enumerate(class_names):
    folder = os.path.join(dataset_dir, label)
    if not os.path.isdir(folder):
        print(f"❌ Skipping {label}, folder not found.")
        continue

    images = [f for f in os.listdir(folder) if f.endswith('.png')]
    if len(images) == 0:
        print(f"⚠️ No images in folder: {label}")
        continue

    sample_images = random.sample(images, min(5, len(images)))
    for img_name in sample_images:
        img_path = os.path.join(folder, img_name)
        pred_idx = predict_image(img_path)
        y_true.append(label_idx)
        y_pred.append(pred_idx)
        print(f"🖼️ {img_name} | Actual: {label} | Predicted: {class_names[pred_idx]}")

# Metrics
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("🧩 Confusion Matrix:")
print(cm)

# --- Start of Alteration ---

print("\n📈 Individual and Mean Accuracy:")
# Calculate accuracy for each class
per_class_accuracy = {}
total_samples = cm.sum()

for i, class_name in enumerate(class_names):
    # True Positives
    tp = cm[i, i]
    # False Positives
    fp = cm[:, i].sum() - tp
    # False Negatives
    fn = cm[i, :].sum() - tp
    # True Negatives
    tn = total_samples - (tp + fp + fn)
    
    # Per-class accuracy
    accuracy = (tp + tn) / total_samples
    per_class_accuracy[class_name] = accuracy
    print(f"- Accuracy for {class_name}: {accuracy:.4f}")

# Calculate mean accuracy
mean_accuracy = np.diag(cm).sum() / total_samples
print(f"- Mean Accuracy: {mean_accuracy:.4f}")

# --- End of Alteration ---
