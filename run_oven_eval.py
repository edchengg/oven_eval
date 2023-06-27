import os
from oven_eval import evaluate_oven_full, load_jsonl

ref_path = "oven"
pred_path = "predictions"

ref_query_val = load_jsonl(os.path.join(ref_path, "oven_query_val.jsonl"))
ref_entity_val = load_jsonl(os.path.join(ref_path, "oven_entity_val.jsonl"))

pred_query_val = load_jsonl(os.path.join(pred_path, "zeroshot_blip2_t5_pretrain_flant5xxl_val_query_bm25.jsonl"))
pred_entity_val = load_jsonl(os.path.join(pred_path, "zeroshot_blip2_t5_pretrain_flant5xxl_val_entity_bm25.jsonl"))

# Note - prediction file format:
# {"data_id": "", "pred_entity_id": ""}
# pred_entity_id is the entity_id from Wikidata (e.g., Q31 is Belgium)

pred_query_val_formal = [{"data_id": line["data_id"], "pred_entity_id": line["bm25"][0][1]} for line in pred_query_val]
pred_entity_val_formal = [{"data_id": line["data_id"], "pred_entity_id": line["bm25"][0][1]} for line in pred_entity_val]

blip2_zeroshot_oven_val = evaluate_oven_full(ref_query_val, ref_entity_val, pred_query_val_formal, pred_entity_val_formal)
print("===== BLIP2 Zeroshot ====")
print("===== Validation ========")
print("===== Final score {}".format(blip2_zeroshot_oven_val["final_score"]))
print("===== Query Split score {}".format(blip2_zeroshot_oven_val["query_score"]))
print("===== Entity Split score {}".format(blip2_zeroshot_oven_val["entity_score"]))
print("===== Query Seen Accuracy {}".format(blip2_zeroshot_oven_val["query_result"]["seen"]))
print("===== Query Unseen Accuracy {}".format(blip2_zeroshot_oven_val["query_result"]["unseen"]))
print("===== Entity Seen Accuracy {}".format(blip2_zeroshot_oven_val["entity_result"]["seen"]))
print("===== Entity Unseen Accuracy {}".format(blip2_zeroshot_oven_val["entity_result"]["unseen"]))
"""
===== BLIP2 Zeroshot ====
===== Validation ========
===== Final score 7.87
===== Query Split score 20.58
===== Entity Split score 4.87
===== Query Seen Accuracy 24.63
===== Query Seen Accuracy 17.68
===== Entity Seen Accuracy 8.55
===== Entity Seen Accuracy 3.4
"""