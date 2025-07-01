import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
import shutil


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=False)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("cnn.pth", map_location=device))
model.eval()

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        module.track_running_stats = False

dummy_input = torch.randn(1, 1, 28, 28).to(device)

try:
    torch.onnx.export(
        model,
        dummy_input,
        "mnist_cnn.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=7,
        do_constant_folding=True,
        verbose=True
    )
    print("Exportation ONNX réussie !")
except Exception as e:
    print(f"Erreur lors de l'exportation : {e}")
    exit(1)

onnx_model = onnx.load("mnist_cnn.onnx")
onnx.checker.check_model(onnx_model)
print("Vérification ONNX réussie !")

session = ort.InferenceSession("mnist_cnn.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
test_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
outputs = session.run([output_name], {input_name: test_input})[0]
print(f"Test d'inférence réussi, forme de la sortie : {outputs.shape}")

shutil.copy("mnist_cnn.onnx", "/mnist_cnn.onnx")
print("Fichier copié dans app/public/")
