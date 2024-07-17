import os
import nibabel as nib


class Config:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join("/dcs", "large", "u5531613", "data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.COLLECTION_DIR = os.path.join(self.BASE_DIR, "collections")
        self.CHECK_POINT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoints')
        self.RES_DIR = os.path.join(self.DATA_DIR, "res")
        self.VISUAL_DIR = os.path.join(self.DATA_DIR, "visual")
        self.PIC_DIR = os.path.join(self.BASE_DIR, "analysis")


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

        example_path = os.path.join(self.BRATS_TRAIN["dir"], "BraTS-PED-00002-000", "BraTS-PED-00002-000-seg.nii.gz")

        self.modalities = ('t2f', 't1c', 't1n', 't2w')
        example = nib.load(example_path)
        self.affine = example.affine


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

        example_path = os.path.join(self.BRATS_TRAIN["dir"], "HGG", 'BraTS19_2013_12_1', "BraTS19_2013_12_1_seg.nii.gz")

        self.modalities = ('t2f', 't1c', 't1n', 't2w')
        example = nib.load(example_path)
        self.affine = example.affine

if __name__ == "__main__":
    config = PediatricConfig()
    print(config.BRATS_TRAIN)
