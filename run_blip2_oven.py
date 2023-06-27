"""BLIP2 zeroshot OVEN inference script."""
import os
import json
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import time

def load_and_process_image(item):
    # Load and preprocess the image
    raw_image = Image.open(id2path[item["image_id"]]).convert("RGB")        
    processed_image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return processed_image, item["question"], item["data_id"]

def process_images_in_batches(batch_data, batch_size, prompt):
    # Create a pool of workers
    # Monitor the progress of the pool
    
    output = []
    print("Generate predictions...")
    # Process images in batches
    for idx, i in enumerate(range(0, len(batch_data), batch_size)):
        if (idx + 1) % 100 == 0:
            print(f"Processing batch {idx}/{len(batch_data)/batch_size}")
        # Subset results for the current batch
        batch_subset = batch_data[i:i+batch_size]

        # Separate the images, questions, and ids
        batch_images, batch_questions, batch_ids = [], [], []

        # Load and preprocess the images
        for item in batch_subset:
            tmp_img, tmp_q, tmp_id = load_and_process_image(item)
            batch_images.append(tmp_img)
            batch_questions.append(tmp_q)
            batch_ids.append(tmp_id)

        # Concatenate the batch images
        image_batch = torch.cat(batch_images, dim=0)
        
        # add prompt to questions
        batch_questions = [prompt.format(q) for q in batch_questions]
        # Generate predictions for the batch
        start_time = time.time()
        answers = model.generate({"image": image_batch, "prompt": batch_questions},
                                 length_penalty=-1)
        print(f"Time for batch {idx}: {time.time() - start_time}")
        for idx, ans in zip(batch_ids, answers):
            output.append({"data_id": idx, "prediction": ans})
    return output

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val_entity", help="val_entity, val_query, test_entity, test_query, or human")
    parser.add_argument("--model_name", type=str, default="blip2_t5", help="blip2_t5 | blip2_vicuna_instruct | blip2_t5_instruct")
    parser.add_argument("--model_type", type=str, default="pretrain_flant5xxl", help="pretrain_flant5xxl | vicuna13b | flant5xxl")
    parser.add_argument("--output_dir", type=str, default="oven_predictions", help="output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")

    args = parser.parse_args()

    split2data = {
        "val_entity": "oven/oven_entity_val.jsonl",
        "val_query": "oven/oven_query_val.jsonl",
        "test_entity": "oven/oven_entity_test.jsonl",
        "test_query": "oven/oven_query_test.jsonl",
        "human": "oven/oven_human.jsonl",
    }

    id2path = dict()

    # load image paths: please prepare a file to map image_id to the actual image_path
    with open("id2image.jsonl", "r") as f:
        for line in f:
            line = json.loads(line)
            image_id = line["image_id"]
            path = line["image_path"]
            id2path[image_id] = path

    # Read the input JSONL file
    with open(split2data[args.split], 'r') as f:
        batch_data = [json.loads(line) for line in f]

    # double check data exists:
    not_exist = []
    clean_batch_data = []
    for idx, item in enumerate(batch_data):
        if idx % 10000 == 0:
            print(f"Processing {idx}/{len(batch_data)}")
        path = id2path[item["image_id"]]
        # check path exists
        if not os.path.exists(path):
            not_exist.append(item["image_id"])
        else:
            clean_batch_data.append(item)
    print(len(not_exist))
    
    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print("Load pretrained model...")
    # loads BLIP-2 pre-trained model
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, 
                                                         model_type=args.model_type, 
                                                         is_eval=True, device=device)
            
    # Desired batch size
    batch_size = args.batch_size

    PROMPT = "Question: {} Short answer:"

    # Run the batch processing function
    output = process_images_in_batches(clean_batch_data, batch_size, prompt=PROMPT)

    # save output into jsonl
    with open(os.path.join(args.output_dir, "zeroshot_{}_{}_{}.jsonl".format(
                args.model_name, args.model_type, args.split
                )), 'w') as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")