import os
import random

input_manifest = os.path.join(os.path.dirname(__file__), '../manifest/train-clean-100-360-500_manifest.json')
train_manifest = os.path.join(os.path.dirname(__file__), '../manifest/train_460-500_manifest.json')
val_manifest = os.path.join(os.path.dirname(__file__), '../manifest/val_460-500_manifest.json')

# Read all lines
with open(input_manifest, 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.shuffle(lines)

n_total = len(lines)
n_val = int(n_total * 0.1)
n_train = n_total - n_val

train_lines = lines[:n_train]
val_lines = lines[n_train:]

with open(train_manifest, 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open(val_manifest, 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print(f"Total: {n_total}, Train: {n_train}, Val: {n_val}")
print(f"Train manifest: {train_manifest}")
print(f"Val manifest: {val_manifest}")
