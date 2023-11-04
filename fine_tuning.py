from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import DefaultDataCollator
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
from evaluate import load
import os
from PIL import ImageFile
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = './dataset/mushroom-classification' # data path to the dataset
model_name = "microsoft/resnet-50"   # model path
train_size=0.8   # test datset size
# evaluate metrics
accuracy = load("accuracy") 
f1 = load("f1")

# Load the mushroom dataset and get the dataset labels
raw_mushroom = load_dataset("imagefolder", data_dir=data_dir)

labels = raw_mushroom["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

# Get 10% of each mushroom species in dataset for the eval dataset
start = 0
for i in os.listdir(os.path.join(data_dir,"train")):
    last_images_index = len(os.listdir(data_dir+"/train/"+i)) + start
    pivot = math.floor((last_images_index-start)*train_size)+start
    if start == 0:
        train_mushroom = Dataset.from_dict(raw_mushroom['train'][0:pivot])
        eval_mushroom = Dataset.from_dict(raw_mushroom['train'][pivot:last_images_index])
    else:
        train_mushroom = concatenate_datasets([train_mushroom, Dataset.from_dict(raw_mushroom['train'][start:pivot])])
        eval_mushroom = concatenate_datasets([eval_mushroom, Dataset.from_dict(raw_mushroom['train'][pivot:last_images_index])])
    start = last_images_index

# Shuffle the dataset
train_mushroom = train_mushroom.shuffle(seed=42)
eval_mushroom = eval_mushroom.shuffle(seed=42)

# pre-process the image data
image_processor = AutoImageProcessor.from_pretrained(model_name)

# Nomalize and Size
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

# Transforms
train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

eval_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(examples):
    examples["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    del examples["image"]
    return examples

def preprocess_eval(examples):
    examples["pixel_values"] = [
        eval_transforms(image.convert("RGB")) for image in examples["image"]
    ]
    del examples["image"]
    return examples
        
# Apply the transform to the image in the dataset
train_mushroom.set_transform(preprocess_train)
eval_mushroom.set_transform(preprocess_eval)

# Train the model
model = AutoModelForImageClassification.from_pretrained(
    model_name, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, 
)

args = TrainingArguments(
    "mushroom-classification",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    # gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    # logging_steps=10,
    logging_dir = "log",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# def data_collator(examples):
#     pixel_values = torch.stack([example["pixel_values"] for example in examples])
#     labels = torch.tensor([example["label"] for example in examples])
#     return {"pixel_values": pixel_values, "labels": labels}

data_collator = DefaultDataCollator()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    a = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f = f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": a, "f1": f}

# Train the model
trainer = Trainer(
    model.to(device),
    args,
    train_dataset=train_mushroom,
    eval_dataset=eval_mushroom,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# Save the trained model
train_results = trainer.train()
trainer.save_model("./fine_tuned_model")
# trainer.save_state()

# Get the log result
train_metrics = train_results.metrics
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)

eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
