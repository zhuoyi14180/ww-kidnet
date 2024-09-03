from monai.networks.nets import SwinUNETR


def get_default():
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=4,
        feature_size=48,
        use_checkpoint=True,
    )

    return model
