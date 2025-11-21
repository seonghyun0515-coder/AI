import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import torchvision.models as models

# ----------------------------------------
# 1. 이미지 전처리 및 시각화 함수 정의
# ----------------------------------------

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0), np.array(image)

def show_cam_on_image(img, mask, title=None):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = np.float32(img) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)

    plt.figure(figsize=(5, 5))
    plt.imshow(cam)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# ----------------------------------------
# 2. Grad-CAM 클래스 정의
# ----------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        target = output[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        grad_cam = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam = torch.nn.functional.relu(grad_cam)
        grad_cam = torch.nn.functional.interpolate(grad_cam, size=(224, 224), mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

        return grad_cam

# ----------------------------------------
# 3. 모델 준비 (ResNet18 - pretrained)
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)  # NORMAL vs PNEUMONIA
model.to(device)
model.eval()

# 👉 여기서 model.load_state_dict(...) 호출로 본인 모델 불러올 수 있음 (학습된 경우)

# ----------------------------------------
# 4. 폴더 순회 및 Grad-CAM 실행
# ----------------------------------------

base_dir = "chest_xray_sample"
categories = ["NORMAL", "PNEUMONIA"]

for category in categories:
    folder = os.path.join(base_dir, category)
    for filename in os.listdir(folder):
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            input_tensor, original_img = load_image(image_path)
            input_tensor = input_tensor.to(device)

            gradcam = GradCAM(model, target_layer='layer4')
            cam = gradcam.generate(input_tensor)

            title = f"{category}: {filename}"
            show_cam_on_image(cv2.resize(original_img, (224, 224)), cam, title)
