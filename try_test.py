import nibabel as nib

img = nib.load('/share/project/zhuoyi/datasets/res/unet2d-brats_ped_2023-poly-valid/BraTS-PED-00111-000.nii.gz')
data = img.get_fdata()

x = 94
y = 66
z = 74

label_value = data[x, y, z]

print(f"标签值: {label_value}")
