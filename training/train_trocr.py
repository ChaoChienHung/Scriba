import os
import torch
from config import Settings
from datetime import datetime
from metrics import compute_metrics
from dataset import HandwritingDataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Constants
# ---------
setting = Settings()
CURRENT_TIME = datetime.now()

# Load Pretrained TrOCR
# ---------------------
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", cache_dir="../cache/model")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Check GPU Existence
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Dataset
# ------------
train_dataset = HandwritingDataset(train_df, processor, cache_path="data_cache/train_trocr.pt")
val_dataset = HandwritingDataset(val_df, processor, cache_path="data_cache/val_trocr.pt")


training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join("trocr_cache", CURRENT_TIME, "checkpoints"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="cer",  # you can define CER later
    greater_is_better=False
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(os.path.join("trocr_cache", CURRENT_TIME, "best"))
