import os


class Config:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join("/dcs", "large", "u5531613", "data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "log")
        self.CHECK_POINT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint')
        self.RES_DIR = os.path.join(self.DATA_DIR, "res")
        self.VISUAL_DIR = os.path.join(self.DATA_DIR, "visual")


class PediatricConfig(Config):
    def __init__(self):
        super().__init__()
        self.BRATS_DIR = os.path.join(self.DATA_DIR, "BraTS-PEDs-2023")

        self.BRATS_TRAIN = {
            "dir": os.path.join(self.BRATS_DIR, "ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"),
            "list": "brats-ped-2023-train.txt",
            "has_label": True
        }

        self.BRATS_VALID = {
            "dir": os.path.join(self.BRATS_DIR, "ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData"),
            "list": "brats-ped-2023-valid.txt",
            "has_label": False
        }

        self.modalities = ('t2f', 't1c', 't1n', 't2w')


class AdultConfig(Config):
    def __init__(self, version=2019):
        super().__init__()
        if version == 2019:
            self.BRATS_DIR = os.path.join(self.DATA_DIR, "BraTS-2019")

            self.BRATS_TRAIN = {
                "dir": os.path.join(self.BRATS_DIR, "MICCAI_BraTS_2019_Data_Training"),
                "list": "brats-2019-train.txt",
                "has_label": True
            }

            self.BRATS_VALID = {
                "dir": os.path.join(self.BRATS_DIR, "MICCAI_BraTS_2019_Data_Validation"),
                "list": "brats-2019-valid.txt",
                "has_label": False
            }

        self.modalities = ('flair', 't1ce', 't1', 't2')

if __name__ == "__main__":
    config = PediatricConfig()
    print(config.BRATS_TRAIN)
