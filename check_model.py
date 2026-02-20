import torch

checkpoint = torch.load('/home/wesleiferreira/Documentos/detectar/model/weights/efficientnet_b0_deepfake_v4.pth', map_location='cpu', weights_only=False)

print("="*50)
print("CHECKPOINT V4 INFO")
print("="*50)
print(f"Keys: {list(checkpoint.keys())}")
print(f"Threshold: {checkpoint.get('best_threshold', 'NAO ENCONTRADO')}")
print(f"AUC: {checkpoint.get('best_auc', 'N/A')}")
print(f"Config: {checkpoint.get('config', {})}")
print("="*50)
