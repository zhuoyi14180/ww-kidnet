import json
import matplotlib.pyplot as plt
import os
import numpy as np
from config import Config

config = Config()

save_path = config.PIC_DIR

file_path = os.path.join(save_path, 'valid_stats-transbts-brats_2019-poly.json')

model = "TransBTS"

dataset = "brats_2019"

with open(file_path, 'r') as f:
    data = json.load(f)

names = [item["name"] for item in data]

res = [item["data"] for item in data]


x = np.arange(len(names))
width = 0.6


fig_loss, ax = plt.subplots(figsize=(20, 8))
rects1_1 = ax.bar(x - width/3, [round(metric[1], 3) for metric in res], width/3, label="Class 1 (ET)")
rects2_1 = ax.bar(x, [round(metric[2], 3) for metric in res], width/3, label="Class 2 (ED)")
rects3_1 = ax.bar(x + width/3, [round(metric[3], 3) for metric in res], width/3, label="Class 3 (NET)")

ax.set_ylabel('Dice Score')
ax.set_title('Class Dice Score for Each Sample on Validation Set')
ax.set_xticks(x - 1)
ax.set_xticklabels(names, rotation=45)
ax.legend()

ax.bar_label(rects1_1, padding=3)
ax.bar_label(rects2_1, padding=3)
ax.bar_label(rects3_1, padding=3)

fig_loss.tight_layout()

plt.savefig(os.path.join(save_path, dataset + "-score" + ".png"))