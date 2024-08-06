import json
import matplotlib.pyplot as plt
import os
from config import Config
import numpy as np


class Visual:
    def __init__(self) -> None:
        self.config = Config()
        self.base_dir = self.config.COLLECTION_DIR
        self.save_dir = self.config.PIC_DIR

    
    def _plot_train(self, in_file, out_file, idx, figsize=(10, 5)):
        with open(os.path.join(self.base_dir, in_file), 'r') as json_file:
            data = json.load(json_file)
        ylabel = "Dice Score" if (idx > 0) else "Dice Loss"
        plt.figure(figsize=figsize)
        plt.plot([item[idx] for item in data], linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel(f'{ylabel}')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, out_file))


    def visual_train_shortcut(self, model, dataset, in_files, out_files=None):
        out_files = out_files or [model + "-" + dataset + "-loss" + ".png", 
                                  model + "-" + dataset + "-class1" + ".png", 
                                  model + "-" + dataset + "-class2" + ".png", 
                                  model + "-" + dataset + "-class3" + ".png"
                                  ]
        
        for idx, in_file, out_file in enumerate(zip(in_files, out_files)):
            self._plot_train(in_file, out_file, idx)

    
    def _plot_valid(self, in_files, out_files, models, figsize=(20, 8)):
        names = []
        results = []
        
        for in_file in in_files:
            with open(os.path.join(self.base_dir, in_file), 'r') as json_file:
                data = json.load(json_file)
            names.append([item["name"] for item in data])
            results.append(item["data"] for item in data)
        x = np.arange(len(names[0]))
        width = 0.35
        fig_loss, ax = plt.subplots(figsize=figsize)
        for ()
        for idx, name, result in enumerate(zip(names, results)):
            rects1_1 = ax.bar(x - width/2, [round(metric[0], 4) for metric in result], width, label=models[idx])

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


    def visual_valid_shortcut(self):
        pass
