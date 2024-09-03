import json
import matplotlib.pyplot as plt
import os
from config import Config
import numpy as np


class Visual:
    def __init__(self) -> None:
        self.config = Config()
        self.base_dir = self.config.COLLECTION_DIR
        save_dir = self.config.PIC_DIR
        self.valid_save_dir = os.path.join(save_dir, "valid")
        self.train_save_dir = os.path.join(save_dir, "train")

        if not os.path.exists(self.valid_save_dir):
            os.makedirs(self.valid_save_dir)

        if not os.path.exists(self.train_save_dir):
            os.makedirs(self.train_save_dir)

    
    def _plot_train(self, in_pair, out_file, idx, figsize=(16, 6)):
        in_train, in_valid = in_pair
        with open(os.path.join(self.base_dir, in_train), 'r') as json_file:
            data_train = json.load(json_file)
        with open(os.path.join(self.base_dir, in_valid), 'r') as json_file:
            data_valid = json.load(json_file)
        ylabel = "Dice Score" if (idx > 0) else "Dice Loss"
        plt.figure(figsize=figsize)
        plt.plot([item[idx] for item in data_train], linestyle='-', color='blue')
        plt.plot([item[idx] for item in data_valid], linestyle='--', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel(f'{ylabel}')
        plt.grid(True)
        plt.savefig(os.path.join(self.train_save_dir, out_file))


    def visual_train_shortcut(self, models, dataset, in_files):

        for i, model in enumerate(models):
            out_files = [
                model + "-" + dataset + "-loss" + ".png", 
                model + "-" + dataset + "-class1" + ".png", 
                model + "-" + dataset + "-class2" + ".png", 
                model + "-" + dataset + "-class3" + ".png"
            ]
            
            for j, out_file in enumerate(out_files):
                self._plot_train(in_files[i], out_file, j)
                

    
    def _post_plot(self, type, inst, *args):
        x = np.arange(len(inst[0]))
        for (fig, ax) in args:
            ax.set_ylabel(type)
            ax.set_xticks(x)
            ax.set_xticklabels(inst[0], rotation=45)
            ax.legend(fontsize=16)
            fig.tight_layout()

    
    def _plot_valid(self, in_files, models, out_dir=None, figsize=(12, 6)):
        names = []
        dices = []
        mious = []
        hd95s = []
        
        for in_file in in_files:
            with open(os.path.join(self.base_dir, "valid", in_file), 'r') as json_file:
                data = json.load(json_file)
            names.append([item["name"] for item in data])
            dices.append([item["dice_more"] for item in data])
            mious.append([item["miou"] for item in data])
            hd95s.append([item["hd95"] for item in data])
        
        width = 0.2
        fig_dice_class1, axd1 = plt.subplots(figsize=figsize)
        fig_dice_class2, axd2 = plt.subplots(figsize=figsize)
        fig_dice_class3, axd3 = plt.subplots(figsize=figsize)

        fig_miou_class1, axm1 = plt.subplots(figsize=figsize)
        fig_miou_class2, axm2 = plt.subplots(figsize=figsize)
        fig_miou_class3, axm3 = plt.subplots(figsize=figsize)

        fig_hd95_class1, axh1 = plt.subplots(figsize=figsize)
        fig_hd95_class2, axh2 = plt.subplots(figsize=figsize)
        fig_hd95_class3, axh3 = plt.subplots(figsize=figsize)

        x = np.arange(len(names[0]))

        start_pos = - 3 * width / 2
        for idx, model in enumerate(models):
            rect1 = axd1.bar(x + (start_pos + width * idx), [round(metric[0], 4) for metric in dices[idx]], width, label=model)
            axd1.bar_label(rect1, padding=3)
            rect2 = axd2.bar(x + (start_pos + width * idx), [round(metric[1], 4) for metric in dices[idx]], width, label=model)
            axd2.bar_label(rect2, padding=3)
            rect3 = axd3.bar(x + (start_pos + width * idx), [round(metric[2], 4) for metric in dices[idx]], width, label=model)
            axd3.bar_label(rect3, padding=3)

        self._post_plot("Dice Score", names, (fig_dice_class1, axd1), (fig_dice_class2, axd2), (fig_dice_class3, axd3))


        fig_dice_class1.savefig(os.path.join(out_dir + "-dice-class1" + ".png"))
        fig_dice_class2.savefig(os.path.join(out_dir + "-dice-class2" + ".png"))
        fig_dice_class3.savefig(os.path.join(out_dir + "-dice-class3" + ".png"))


        for idx, model in enumerate(models):
            rect1 = axm1.bar(x + (start_pos + width * idx), [round(metric[3], 4) for metric in mious[idx]], width, label=model)
            axm1.bar_label(rect1, padding=3)
            rect2 = axm2.bar(x + (start_pos + width * idx), [round(metric[4], 4) for metric in mious[idx]], width, label=model)
            axm2.bar_label(rect2, padding=3)
            rect3 = axm3.bar(x + (start_pos + width * idx), [round(metric[5], 4) for metric in mious[idx]], width, label=model)
            axm3.bar_label(rect3, padding=3)

        self._post_plot("mIoU (%)", names, (fig_miou_class1, axm1), (fig_miou_class2, axm2), (fig_miou_class3, axm3))


        fig_miou_class1.savefig(os.path.join(out_dir + "-miou-class1" + ".png"))
        fig_miou_class2.savefig(os.path.join(out_dir + "-miou-class2" + ".png"))
        fig_miou_class3.savefig(os.path.join(out_dir + "-miou-class3" + ".png"))


        for idx, model in enumerate(models):
            rect1 = axh1.bar(x + (start_pos + width * idx), [round(metric[4], 4) for metric in hd95s[idx]], width, label=model)
            axh1.bar_label(rect1, padding=3)
            rect2 = axh2.bar(x + (start_pos + width * idx), [round(metric[5], 4) for metric in hd95s[idx]], width, label=model)
            axh2.bar_label(rect2, padding=3)
            rect3 = axh3.bar(x + (start_pos + width * idx), [round(metric[6], 4) for metric in hd95s[idx]], width, label=model)
            axh3.bar_label(rect3, padding=3)

        self._post_plot("hd95 (mm)", names, (fig_hd95_class1, axh1), (fig_hd95_class2, axh2), (fig_hd95_class3, axh3))


        fig_hd95_class1.savefig(os.path.join(out_dir + "-hd95-class1" + ".png"))
        fig_hd95_class2.savefig(os.path.join(out_dir + "-hd95-class2" + ".png"))
        fig_hd95_class3.savefig(os.path.join(out_dir + "-hd95-class3" + ".png"))


    def visual_valid_shortcut(self, in_files, models, out_dir=None, dataset="brats_pred_2023", figsize=(20, 8)):
        if out_dir is None:
            out_dir = os.path.join(self.valid_save_dir, dataset)
        self._plot_valid(in_files, models, out_dir=out_dir, figsize=(20, 8))


if __name__ == "__main__":
    visual = Visual()
    models = ["swinunetr", "transbts", "kidnet"]

    # in_pairs = (
    #     ("transbts-brats_ped_2023-train.json", "transbts-brats_ped_2023-valid.json"), 
    #     ("unet3d-brats_ped_2023-train.json", "unet3d-brats_ped_2023-valid.json"), 
    #     ("vit2d-brats_ped_2023-train.json", "vit2d-brats_ped_2023-valid.json"), 
    #     ("unet2d-brats_ped_2023-train.json", "unet2d-brats_ped_2023-valid.json"), 
    #     ("swinunetr-brats_ped_2023-train.json", "swinunetr-brats_ped_2023-valid.json")
    # )
    
    # visual.visual_train_shortcut(models, "brats_ped_2023", in_files=in_pairs)

    config = Config()
    
    in_files = [
        "valid_stats-transbts-brats_ped_2023-poly-valid.json", 
        "valid_stats-unet3d-brats_ped_2023-poly-valid.json", 
        # "valid_stats-vit2d-brats_ped_2023-poly-valid.json", 
        "valid_stats-unet2d-brats_ped_2023-poly-valid.json", 
        # "valid_stats-swinunetr-brats_ped_2023-poly-valid.json"
    ]
    visual.visual_valid_shortcut(in_files, models, dataset="brats_ped_2023", figsize=(40, 16))