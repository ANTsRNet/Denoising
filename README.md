# App:  Brain denoising

Deep learning app made for T1-weighted MRI brain denoising using ANTsRNet

## Model training notes

* Training data: IXI, NKI, Kirby, Oasis
* Unet model (see ``Scripts/Training/``).
* Patch-based (32, 32, 32)

## Sample prediction usage

```
#
#  Usage:
#    Rscript doDenoising.R inputFile inputMaskFile outputFile
#
#  MacBook Pro 2016 (no GPU)
#

$ Rscript Scripts/doDenoising.R Data/Example/1097782_defaced_MPRAGE.nii.gz Data/Example/1097782_defaced_MPRAGEBrainExtractionMask.nii.gz denoisedOutput.nii.gz

```

## Sample results

![Brain extraction results](Documentation/Images/resultsDenoising.png)
