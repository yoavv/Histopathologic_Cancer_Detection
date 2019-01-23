# Histopathologic Cancer Detection

## overview
in the axilla are the first place breast cancer is likely to spread. Metastatic involvement of lymph nodes is one of the most important prognostic factors in breast cancer. Prognosis is poorer when cancer has spread to the lymph nodes. This is why lymph nodes are surgically removed and examined microscopically. However, the diagnostic procedure for pathologists is tedious and time-consuming. But most importantly, small metastases are very difficult to detect and sometimes they are missed.
create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans

<img src="images/explore1.png" width=400/>
<img src="images/explore2.png" width=400>
<img src="images/explore3.png" width=400>

In this dataset, you are provided with a large number of small pathology images to classify. Files are named with an image id. The train_labels.csv file provides the ground truth for the images in the train folder. You are predicting the labels for the images in the test folder. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates. We have otherwise maintained the same data and splits as the PCam benchmark.

<img src="images/examples.png" alt="drawing" width="800"/>
