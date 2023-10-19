# Overview
The implementation for the paper DCL-FAS (ICCV2023)

(more to be updated)

# Protocols
The protocols are 


# How to run the code
## Data preparation
Please refer to https://github.com/RizhaoCai/FAS_DataManager to download data and process the dataset.
The loading of data is by reading a data list file (.csv).
As provided in [data_list](data_list), different data list files show different data settings.

Below is a segment extracted from one data list file (csv), 
```
    /home/Dataset/Face_Spoofing/frames/CASIA-FASD/train_release/1/1.avi/10.png,0
    /home/Dataset/Face_Spoofing/frames/CASIA-FASD/train_release/1/1.avi/9.png,0
    /home/Dataset/Face_Spoofing/frames/CASIA-FASD/train_release/1/1.avi/8.png,0
    /home/Dataset/Face_Spoofing/frames/CASIA-FASD/train_release/1/1.avi/7.png,0
    /home/Dataset/Face_Spoofing/frames/CASIA-FASD/train_release/1/1.avi/6.png,0
```

`/home/Dataset/Face_Spoofing/frames` is a path that depends on how you configure your dataset storage. `CASIA-FASD/train_release/1/1.avi` is the name of the example (video), which is identical to this example's original relative path in the folder of the original dataset CASIA-FASD. `CASIA-FASD/train_release/1/1.avi/10.png` means that the 10th frame is extracted from the video example `CASIA-FASD/train_release/1/1.avi`. It should be noticed that `/home/Dataset/Face_Spoofing/frames/CASIA-FASD/train_release/1/1.avi/` is a folder instead of an avi file. The second column indicates the labels. `0` means bona fide (real, genuine face). `1` means print photo attack. `2` means replay attack. `3` means 3D mask attacks. 

# Train the model

# Evaluation

# 

