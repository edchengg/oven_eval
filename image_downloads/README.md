# OVEN Images

## Option 1 (Download images from Snapshot)
- [24/9/1] We have switched data download from Google Drive to Huggingface datasets due to storage limitation issue.
  
To download annotations and an image snapshot, check out Huggingface dataset at 🤗: [ychenNLP/oven](https://huggingface.co/datasets/ychenNLP/oven), which support high-speed wget command.

## Option 2 (Download images from Original Source)
To download all images from the source dataset, please run all download scripts. Then run the following script to merge all data with [ovenid2impath.csv](https://forms.gle/SbWLfbexhQV9w2H26):
```python
python merge_oven_images.py
```
