import SimpleITK as sitk
import numpy as np

def read_and_resample_nrrd(path, out_spacing=(2.0,2.0,2.0), is_label=False):
    img = sitk.ReadImage(path)
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    out_size = [int(round(original_size[i] * (original_spacing[i] / out_spacing[i]))) for i in range(3)]
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(img)
    f.SetOutputSpacing(out_spacing)
    f.SetSize(out_size)
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    res = f.Execute(img)
    arr = sitk.GetArrayFromImage(res)  # (D,H,W)
    return arr, res
