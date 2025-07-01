import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

# Définition du modèle, exactement comme dans ton notebook
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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instanciation du modèle
model = Net()

# Chargement des poids entraînés
model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))

# Passage en mode évaluation (très important pour Dropout, BatchNorm, etc)
model.eval()

# Entrée factice correspondant à la taille attendue (batch=1, channel=1, 28x28)
dummy_input = torch.randn(1, 1, 28, 28)

# Export ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mnist_cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=9,  # React ONNX supporte bien opset 9
    do_constant_folding=True,  # Optimisation
)

# Vérification du modèle exporté
onnx_model = onnx.load("mnist_cnn.onnx")
onnx.checker.check_model(onnx_model)

print("Export ONNX terminé et validé avec succès !")
