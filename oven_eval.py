"""OVEN Evaluation Script."""
import re
import json
import string
from typing import Any, Dict, Generator, List, Tuple, Union
from collections import defaultdict


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of Dictionaries."""
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def prepare_qid2example(
    reference: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
    """Convert reference to qid2example dictionary."""
    qid2example = dict()
    for r in reference:
        qid = r['data_id']
        qid2example[qid] = r
    return qid2example

def evaluate_oven(ref: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate the predicted results against the reference.

    :param ref: a list of dictionaries each representing a reference item.
    :param pred: a list of dictionaries each representing a predicted item.
    :return: a dictionary of accuracy for each split.
    """
    split2res = defaultdict(list)
    qid2example = prepare_qid2example(ref)

    for pred_item in pred:
        data_id = pred_item["data_id"]
        ref_item = qid2example[data_id]
        ref_ent_id = ref_item["entity_id"]
        pred_ent_id = pred_item["pred_entity_id"]
        data_split = ref_item["data_split"]
        
        match_score = int(ref_ent_id == pred_ent_id)
        split2res[data_split].append(match_score)

    result = {}
    for split, results in split2res.items():
        accuracy = round(sum(results) / len(results) * 100, 2)
        if "_seen" in split:
            result["seen"] = accuracy
        elif "_unseen" in split:
            result["unseen"] = accuracy
    
    return result

def harmonic_mean(*args: float) -> float:
    """Calculate the harmonic mean of the input arguments."""
    args_safe = [a if a != 0 else 1e-12 for a in args]
    hmean = len(args_safe) / sum((1.0 / val) for val in args_safe)
    return hmean

def validate_prediction_inputs(predictions: List[Dict[str, Any]]) -> None:
    """
    Validate that all required keys are present in the prediction inputs.

    :param predictions: a list of dictionaries each representing a predicted item.
    :raises ValueError: if a required key is missing from any prediction input.
    """
    for prediction in predictions:
        if "pred_entity_id" not in prediction:
            raise ValueError(f"pred_entity_id is missing in prediction data_id {prediction['data_id']}")


def evaluate_oven_full(ref_query: List[Dict[str, Any]], ref_entity: List[Dict[str, Any]], 
                  pred_query: List[Dict[str, Any]], pred_entity: List[Dict[str, Any]]) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate the final result based on both query and entity results.

    :param ref_query: a list of dictionaries each representing a reference query.
    :param ref_entity: a list of dictionaries each representing a reference entity.
    :param pred_query: a list of dictionaries each representing a predicted query.
    :param pred_entity: a list of dictionaries each representing a predicted entity.
    :return: a dictionary containing calculated scores and results.
    """
    validate_prediction_inputs(pred_query)
    validate_prediction_inputs(pred_entity)

    #  prepare qid2example
    query_result = evaluate_oven(ref_query, pred_query)
    entity_result = evaluate_oven(ref_entity, pred_entity)

    final_result = {}
    query_score = harmonic_mean(query_result["seen"], query_result["unseen"])
    entity_score = harmonic_mean(entity_result["seen"], entity_result["unseen"])
    final_score = harmonic_mean(query_score, entity_score)

    final_result = {
        "query_score": round(query_score, 2),
        "entity_score": round(entity_score, 2),
        "final_score": round(final_score, 2),
        "query_result": query_result,
        "entity_result": entity_result
    }
    return final_result
