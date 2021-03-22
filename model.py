import torch
import torch.nn as nn 
from resnest.torch.resnet import ResNet, Bottleneck

MODEL_CONFIGS = {
    "resnest50_fast_1s1x64d":
    {
        "num_classes": 264,
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "radix": 1,
        "groups": 1,
        "bottleneck_width": 64,
        "deep_stem": True,
        "stem_width": 32,
        "avg_down": True,
        "avd": True,
        "avd_first": True
    }
}

def create_model(n_class=51):
    model = ResNet(**MODEL_CONFIGS["resnest50_fast_1s1x64d"])
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_class)
    return model

def load_weights(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model

def get_model(path, n_class=51):
    model = create_model(n_class=n_class)
    model = load_weights(model, path)
    return model.eval()