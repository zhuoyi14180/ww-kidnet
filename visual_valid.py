import json
import matplotlib.pyplot as plt
import os
import numpy as np

file_path_1 = os.path.join('./pic', 'valid_stats-transbts-brats_ped_2023_poly.json')
file_path_2 = os.path.join('./pic', 'valid_stats-transbts-brats_ped_2023.json')

save_path = './pic'

model_1 = "transbts_plly"
model_2 = "transbts"

dataset = "brats_ped_2023"

with open(file_path_1, 'r') as f:
    data_1 = json.load(f)

with open(file_path_2, 'r') as f:
    data_2 = json.load(f)

names = [item["name"] for item in data_1]

res_1 = [item["data"] for item in data_1]
res_2 = [item["data"] for item in data_2]


x = np.arange(len(names))
width = 0.35


fig_loss, ax = plt.subplots(figsize=(20, 8))
rects1_1 = ax.bar(x - width/2, [round(metric[0], 4) for metric in res_1], width, label='TransBTS')
rects2_1 = ax.bar(x + width/2, [round(metric[0], 4)  for metric in res_2], width, label='3D UNet')

# 添加一些文本标签
ax.set_ylabel('Dice Loss')
ax.set_title('Dice Loss for Each Sample on Validation Set')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.legend()

ax.bar_label(rects1_1, padding=3)
ax.bar_label(rects2_1, padding=3)

fig_loss.tight_layout()

plt.savefig(os.path.join(save_path, dataset + "-loss" + ".png"))


fig_class1, ax = plt.subplots()
rects1_2 = ax.bar(x - width/2, [round(metric[1], 4) for metric in res_1], width, label='TransBTS')
rects2_2 = ax.bar(x + width/2, [round(metric[1], 4) for metric in res_2], width, label='3D UNet')

# 添加一些文本标签
ax.set_ylabel('Dice Score')
ax.set_title('Dice Score for Class 1 on Validation Set')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.legend()

ax.bar_label(rects1_2, padding=3)
ax.bar_label(rects2_2, padding=3)

fig_class1.tight_layout()

plt.savefig(os.path.join(save_path, dataset + "-class1" + ".png"))



fig_class2, ax = plt.subplots()
rects1_3 = ax.bar(x - width/2, [round(metric[2], 4) for metric in res_1], width, label='TransBTS')
rects2_3 = ax.bar(x + width/2, [round(metric[2], 4) for metric in res_2], width, label='3D UNet')

# 添加一些文本标签
ax.set_ylabel('Dice Score')
ax.set_title('Dice Score for Class 2 on Validation Set')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.legend()

ax.bar_label(rects1_3, padding=3)
ax.bar_label(rects2_3, padding=3)

fig_class2.tight_layout()

plt.savefig(os.path.join(save_path, dataset + "-class2" + ".png"))



fig_class3, ax = plt.subplots()
rects1_4 = ax.bar(x - width/2, [round(metric[3], 4) for metric in res_1], width, label='TransBTS')
rects2_4 = ax.bar(x + width/2, [round(metric[3], 4) for metric in res_2], width, label='3D UNet')

ax.set_ylabel('Dice Score')
ax.set_title('Dice Score for Class 3 on Validation Set')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45)
ax.legend()

ax.bar_label(rects1_4, padding=3)
ax.bar_label(rects2_4, padding=3)

fig_class3.tight_layout()

plt.savefig(os.path.join(save_path, dataset + "-class3" + ".png"))