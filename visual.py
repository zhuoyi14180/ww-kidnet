import json
import matplotlib.pyplot as plt
import os

file_path = 'unet3d-brats_ped_2023.json'

save_path = './pic'

model = "unet3d"

dataset = "brats_ped_2023"

with open(file_path, 'r') as f:
    data = json.load(f)


plt.figure(figsize=(10, 5))
plt.plot([item[0] for item in data], linestyle='-', color='b')
plt.title('Train Loss for TransBTS on BraTS-PEDs 2023')
plt.xlabel('Epoch')
plt.ylabel('Dice Loss')
plt.grid(True)

plt.savefig(os.path.join(save_path, model + "-" + dataset + "-loss" + ".png"))


plt.figure(figsize=(10, 5))
plt.plot([item[1] for item in data], linestyle='-', color='darkorange')
plt.title('Dice Score for Class 1 by TransBTS on BraTS-PEDs 2023')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.grid(True)

plt.savefig(os.path.join(save_path, model + "-" + dataset + "-class1" + ".png"))


plt.figure(figsize=(10, 5))
plt.plot([item[2] for item in data], linestyle='-', color='darkorange')
plt.title('Dice Score for Class 2 by TransBTS on BraTS-PEDs 2023')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.grid(True)

plt.savefig(os.path.join(save_path, model + "-" + dataset + "-class2" + ".png"))


plt.figure(figsize=(10, 5))
plt.plot([item[3] for item in data], linestyle='-', color='darkorange')
plt.title('Dice Score for Class 3 by TransBTS on BraTS-PEDs 2023')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.grid(True)

plt.savefig(os.path.join(save_path, model + "-" + dataset + "-class3" + ".png"))
