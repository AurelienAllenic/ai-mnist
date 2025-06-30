import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

# J'utilise la même class que dans mon fichier cnn.ipynb
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x

# Instancier le modèle et charger les poids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("cnn.pth", map_location=device))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28).to(device)

# Exporter le modèle
torch.onnx.export(
    model,
    dummy_input,
    "mnist_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

onnx_model = onnx.load("mnist_cnn.onnx")
onnx.checker.check_model(onnx_model)
print("Modèle ONNX exporté avec succès dans 'mnist_cnn.onnx' !")