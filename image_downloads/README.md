# OVEN Images

## Option 1 (Download images from Snapshot)
To download annotations and an image snapshot (shard00-05.tar), please fill the [form](https://forms.gle/SbWLfbexhQV9w2H26).

## Option 2 (Download images from Original Source)
To download all images from the source dataset, please run all download scripts. Then run the following script to merge all data with [ovenid2impath.csv](https://drive.google.com/file/d/15ICSQfyF-lwpqYjkXZ3DryY4FVimQW0b/view?usp=drive_link):
```python
python merge_oven_images.py
```