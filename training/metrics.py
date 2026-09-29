from datasets import load_metric

cer_metric = load_metric("cer")

def compute_metrics(prediction):
    pred_ids = prediction.predictions
    labels_ids = prediction.label_ids
    preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=preds, references=labels)
    return {"cer": cer}
