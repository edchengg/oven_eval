# OVEN Images

## Option 1 (Download images from Snapshot)
To download annotations and an image snapshot (shard00-05.tar):
| Data file name | Size |
| --- | ---: |
| [image snapshot](https://drive.google.com/drive/folders/1gPUPo2q7JKeTf6l_aJfU7k8NMmyskPeB?usp=drive_link) | 239 GB |

## Option 2 (Download images from Original Source)
To download all images from the source dataset, please run all download scripts. Then run the following script to merge all data with [ovenid2impath.csv](https://drive.google.com/file/d/15ICSQfyF-lwpqYjkXZ3DryY4FVimQW0b/view?usp=drive_link):
```python
python merge_oven_images.py
```