import torch
import os

path = "deepfake_model.pt"

if not os.path.exists(path):
    print("Model file not found!")
else:
    print(f"Model size: {os.path.getsize(path) / (1024 * 1024):.2f} MB")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("\nModel checkpoint loaded successfully!")
    print("Checkpoint type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Number of saved entries:", len(checkpoint))
        print("First keys:", list(checkpoint.keys())[:10])