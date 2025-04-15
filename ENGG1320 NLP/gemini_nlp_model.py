#hippity hoppity get off my property 
#skill issue lmao
from datasets import load_dataset, Features, Value, ClassLabel
import numpy as np # Import numpy if not already imported

# --- FIX START ---
# Step 1: Load temporarily or peek to find unique labels
temp_intent_dataset = load_dataset('csv', data_files={'train': 'intents_train.csv'})
unique_labels = sorted(list(set(temp_intent_dataset['train']['label'])))
print(f"Found labels for features: {unique_labels}")

# Step 2: Define the features, explicitly setting 'label' as ClassLabel
intent_features = Features({
    'text': Value('string'),  # Assuming 'text' is the column name for the message
    'label': ClassLabel(names=unique_labels) # Explicitly define 'label'
})

# Step 3: Reload the dataset using the defined features
intent_dataset = load_dataset('csv',
                              data_files={'train': 'intents_train.csv', 'test': 'intents_test.csv'},
                              features=intent_features)

# Now you can access .names because the feature type is correctly set
intent_labels = intent_dataset['train'].features['label'].names
intent_label2id = {label: i for i, label in enumerate(intent_labels)}
id2intent_label = {i: label for i, label in enumerate(intent_labels)}
num_intent_labels = len(intent_labels)
# --- FIX END ---

# For NER (assuming JSON Lines)
ner_dataset = load_dataset('json', data_files={'train': 'ner_train.jsonl', 'test': 'ner_test.jsonl'})
# Assuming ner_tags are strings in the JSONL file, create mappings
# Flatten the list of lists of tags, get unique tags, sort them
ner_tags_list = list(set([tag for example in ner_dataset['train'] for tag in example['ner_tags']]))
ner_tags_list.sort() # Important for consistency
ner_label2id = {label: i for i, label in enumerate(ner_tags_list)}
id2ner_label = {i: label for i, label in enumerate(ner_tags_list)}
num_ner_labels = len(ner_tags_list)

print("Intent Labels Mapping:", intent_label2id)
print("Number of Intent Labels:", num_intent_labels)
print("NER Labels:", ner_label2id)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch

model_checkpoint = "distilbert-base-uncased" # Or another DistilBERT variant

# --- Intent Classification Setup ---
intent_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
intent_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_intent_labels,
    id2label=id2intent_label,
    label2id=intent_label2id
)

# --- NER Setup ---
ner_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # Can often reuse the same tokenizer
ner_model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_ner_labels,
    id2label=id2ner_label,
    label2id=ner_label2id
)

def tokenize_intents(batch):
    # Tokenize text
    tokenized_inputs = intent_tokenizer(
        batch['text'],
        truncation=True,
        padding='max_length',
        max_length=128 # Adjust max_length if needed
    )
    # The labels are likely already integers, so just assign them directly
    tokenized_inputs["labels"] = batch['label'] # <-- Use the label directly
    return tokenized_inputs

# Now run the map function again
encoded_intent_dataset = intent_dataset.map(tokenize_intents, batched=True)

# You can optionally print to verify
print("Sample encoded intent data:", encoded_intent_dataset['train'][0])

def tokenize_and_align_labels_ner(examples):
    tokenized_inputs = ner_tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding='max_length', max_length=128) # is_split_into_words=True is important

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]): # Assumes ner_tags column holds the list of tags
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to original words
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: # Special tokens like [CLS], [SEP]
                label_ids.append(-100)
            elif word_idx != previous_word_idx: # First token of a new word
                label_ids.append(ner_label2id[label[word_idx]])
            else: # Subsequent tokens of the same word
                # Option 1: Assign -100 (common practice)
                label_ids.append(-100)
                # Option 2: Assign the same label (e.g., I-TAG if the first was B-TAG/I-TAG) - less common for BERT-like models
                # label_ids.append(ner_label2id[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

encoded_ner_dataset = ner_dataset.map(tokenize_and_align_labels_ner, batched=True)

# --- Intent Training Args ---
intent_training_args = TrainingArguments(
    output_dir='./results/intent_classifier',
    num_train_epochs=3, # Start with 3-5, adjust based on validation performance
    per_device_train_batch_size=16, # Adjust based on GPU memory
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs/intent_classifier',
    logging_steps=10,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",       # Save model checkpoint at the end of each epoch
    load_best_model_at_end=True, # Load the best performing model (on validation set)
    metric_for_best_model="accuracy", # Or f1, precision, recall
    push_to_hub=False, # Set to True if you want to upload to Hugging Face Hub
)

# --- NER Training Args ---
ner_training_args = TrainingArguments(
    output_dir='./results/ner_model',
    num_train_epochs=5, # NER often benefits from slightly more epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs/ner_model',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1", # Use f1 for NER evaluation
    push_to_hub=False,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics_intent(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted') # Use 'weighted' or 'macro' for multi-class
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from seqeval.metrics import classification_report, f1_score

def compute_metrics_ner(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) and convert IDs to labels
    true_predictions = [
        [id2ner_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2ner_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Use seqeval's f1_score for the primary metric Trainer uses
    f1 = f1_score(true_labels, true_predictions, average="weighted") # Or 'macro' or 'micro'
    # You can also get a detailed report
    report = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": f1,
        # You could add overall accuracy here too if desired
    }

# --- Intent Trainer ---
intent_trainer = Trainer(
    model=intent_model,
    args=intent_training_args,
    train_dataset=encoded_intent_dataset["train"],
    eval_dataset=encoded_intent_dataset["test"], # Use test set here, ideally should be a validation set
    tokenizer=intent_tokenizer,
    compute_metrics=compute_metrics_intent,
)

# --- NER Trainer ---
# Need data collator for padding NER batches correctly
from transformers import DataCollatorForTokenClassification
ner_data_collator = DataCollatorForTokenClassification(tokenizer=ner_tokenizer)

ner_trainer = Trainer(
    model=ner_model,
    args=ner_training_args,
    train_dataset=encoded_ner_dataset["train"],
    eval_dataset=encoded_ner_dataset["test"], # Ideally a validation set
    tokenizer=ner_tokenizer,
    data_collator=ner_data_collator, # Add the data collator
    compute_metrics=compute_metrics_ner,
)

# --- Start Training ---
print("Starting Intent Model Training...")
intent_trainer.train()

print("\nStarting NER Model Training...")
ner_trainer.train()

# --- Save the final models ---
intent_trainer.save_model("./final_model/intent_classifier")
ner_trainer.save_model("./final_model/ner_model")
intent_tokenizer.save_pretrained("./final_model/intent_classifier")
ner_tokenizer.save_pretrained("./final_model/ner_model")

# (Optional) Copy to Google Drive if mounted
# !cp -r ./final_model /content/drive/MyDrive/my_smart_calendar_models

# Ensure you have your encoded test datasets ready
# encoded_intent_dataset['test']
# encoded_ner_dataset['test'] # Make sure this was created similarly to the intent one

print("Evaluating Intent Model on Test Set:")
intent_eval_results = intent_trainer.evaluate(encoded_intent_dataset['test'])
print(intent_eval_results)

print("\nEvaluating NER Model on Test Set:")
ner_eval_results = ner_trainer.evaluate(encoded_ner_dataset['test'])
print(ner_eval_results)

# Define final save directories
final_intent_model_dir = "./final_model/intent_model"
final_ner_model_dir = "./final_model/ner_model" # As used before

# Save Intent Model & Tokenizer
intent_trainer.save_model(final_intent_model_dir)
intent_tokenizer.save_pretrained(final_intent_model_dir)
print(f"Intent model and tokenizer saved to {final_intent_model_dir}")

# Save NER Model & Tokenizer (Trainer might have already saved the model)
ner_trainer.save_model(final_ner_model_dir)
ner_tokenizer.save_pretrained(final_ner_model_dir) # Ensures tokenizer is saved too
print(f"NER model and tokenizer saved to {final_ner_model_dir}")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import numpy as np

# Load the fine-tuned models and tokenizers from your saved directory
intent_tokenizer = AutoTokenizer.from_pretrained(final_intent_model_dir)
intent_model = AutoModelForSequenceClassification.from_pretrained(final_intent_model_dir)

ner_tokenizer = AutoTokenizer.from_pretrained(final_ner_model_dir)
ner_model = AutoModelForTokenClassification.from_pretrained(final_ner_model_dir)

# Make sure you have your label mappings available (defined earlier)
# id2intent_label = {0: 'Add', 1: 'Edit', 2: 'Remove'} # Example
# id2ner_label = {0: 'O', 1: 'B-DESC', 2: 'I-DESC', ...} # Example

def predict_intent_ner(text):
    # --- Intent Prediction ---
    intent_inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        intent_logits = intent_model(**intent_inputs).logits
    intent_probabilities = torch.softmax(intent_logits, dim=-1).squeeze()
    predicted_intent_id = torch.argmax(intent_probabilities).item()
    predicted_intent = id2intent_label[predicted_intent_id]
    intent_confidence = intent_probabilities[predicted_intent_id].item()

    # --- NER Prediction ---
    ner_inputs = ner_tokenizer(text, return_tensors="pt", truncation=True) # Don't pad here yet
    tokens = ner_tokenizer.convert_ids_to_tokens(ner_inputs["input_ids"].squeeze().tolist())
    # Ensure tokens don't include special tokens like [CLS], [SEP] for NER processing if tokenizer adds them automatically
    # This might need adjustment based on your specific tokenizer's behavior
    # A common approach is to get offsets and map back to original words later

    with torch.no_grad():
        ner_logits = ner_model(**ner_inputs).logits

    predicted_ner_ids = torch.argmax(ner_logits, dim=-1).squeeze().tolist()

    # --- NER Post-processing (Crucial Step) ---
    # Align tokens with predicted tags and group B-/I- tags
    entities = []
    current_entity_tokens = []
    current_entity_tag = None

    # Get word IDs to handle subword tokenization (e.g., "meeting" -> "meet", "##ing")
    word_ids = ner_inputs.word_ids()
    previous_word_id = None

    for i, token_id in enumerate(predicted_ner_ids):
        # Ignore special tokens (where word_id is None)
        current_word_id = word_ids[i]
        if current_word_id is None:
            continue

        predicted_tag = id2ner_label[token_id]

        # Only process the first subword token of a word
        if current_word_id != previous_word_id:
            # If we were tracking an entity, store it
            if current_entity_tag and current_entity_tokens:
                 entity_text = ner_tokenizer.convert_tokens_to_string(current_entity_tokens)
                 entities.append({"text": entity_text.strip(), "type": current_entity_tag})
                 current_entity_tokens = []
                 current_entity_tag = None

            # Start a new entity if tag is B-
            if predicted_tag.startswith("B-"):
                current_entity_tokens.append(tokens[i])
                current_entity_tag = predicted_tag[2:] # Get tag type (e.g., DESC)
            # Reset if tag is O
            elif predicted_tag == "O":
                 pass # Do nothing, not part of an entity

        # If it's a subsequent subword token
        else:
             # Continue current entity if tag is I- and matches current type
             if predicted_tag.startswith("I-") and current_entity_tag == predicted_tag[2:]:
                 current_entity_tokens.append(tokens[i])
             # Otherwise (e.g., O tag within a word, mismatch I- tag), the entity ends here
             elif current_entity_tag and current_entity_tokens:
                 entity_text = ner_tokenizer.convert_tokens_to_string(current_entity_tokens)
                 entities.append({"text": entity_text.strip(), "type": current_entity_tag})
                 current_entity_tokens = []
                 current_entity_tag = None


        previous_word_id = current_word_id

    # Add any trailing entity
    if current_entity_tag and current_entity_tokens:
        entity_text = ner_tokenizer.convert_tokens_to_string(current_entity_tokens)
        entities.append({"text": entity_text.strip(), "type": current_entity_tag})

    # Alternative NER Post-processing using Hugging Face pipeline (simpler):
    # from transformers import pipeline
    # ner_pipeline = pipeline("token-classification", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple") # Or other strategies
    # raw_entities = ner_pipeline(text)
    # entities = [{"text": entity['word'], "type": entity['entity_group']} for entity in raw_entities]


    return {
        "text": text,
        "intent": {"label": predicted_intent, "confidence": intent_confidence},
        "entities": entities
    }

import json
import re
from transformers import pipeline
import os # Import os for file existence checks

# --- 1. Define Model Paths ---
# Ensure these paths correctly point to where your models are saved locally
intent_model_path = "./final_model/intent_classifier"
ner_model_path = "./final_model/ner_model"

# --- Check if model directories exist ---
if not os.path.isdir(intent_model_path):
    print(f"Error: Intent model directory not found at '{intent_model_path}'")
    exit() # Stop execution if models aren't found
if not os.path.isdir(ner_model_path):
    print(f"Error: NER model directory not found at '{ner_model_path}'")
    exit()

# --- 2. Load Pipelines (Do this once at the start) ---
try:
    print("Loading intent classification pipeline...")
    # Specify device=0 to try using GPU 0 if available, otherwise it defaults to CPU
    intent_classifier = pipeline(
        "text-classification",
        model=intent_model_path,
        tokenizer=intent_model_path,
        device=0 if torch.cuda.is_available() else -1 # Use GPU 0 if available, else CPU
    )
    print("Loading NER pipeline...")
    ner_pipeline = pipeline(
        "token-classification",
        model=ner_model_path,
        tokenizer=ner_model_path,
        aggregation_strategy="simple", # Groups B-TAG/I-TAG
        device=0 if torch.cuda.is_available() else -1 # Use GPU 0 if available, else CPU
    )
    print("Pipelines loaded successfully.")
    # Check which device is being used (optional)
    print(f"Intent classifier running on: {intent_classifier.device}")
    print(f"NER pipeline running on: {ner_pipeline.device}")

except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure the model paths are correct, models are downloaded,")
    print("and required libraries (like torch, transformers) are installed.")
    exit()

# --- 3. Define the Parsing Function (Keep this as before) ---
def parse_user_message(message):
    """
    Parses a user message to extract intent and entities using pre-loaded pipelines.
    """
    print(f"\nProcessing message: '{message}'")
    # 1. Predict Intent
    intent_result = intent_classifier(message)[0] # Get the top prediction
    intent = intent_result['label']
    intent_confidence = intent_result['score']
    print(f"  Intent: {intent} (Confidence: {intent_confidence:.4f})")

    # 2. Predict Entities
    ner_results = ner_pipeline(message)
    print(f"  Raw NER results: {ner_results}")

    # 3. Process & Structure Entities
    entities = {}
    for entity in ner_results:
        entity_type = entity['entity_group']
        entity_value = entity['word'].strip()
        # Simple cleanup (can be expanded)
        entity_value = re.sub(r' ##', '', entity_value) # Remove subword artifacts if aggregation didn't handle perfectly
        entity_value = re.sub(r'\s+', ' ', entity_value).strip() # Normalize whitespace

        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append({
            "text": entity_value,
            "confidence": entity['score'],
            # You might want start/end character indices from ner_result if needed
            # "start": entity['start'],
            # "end": entity['end'],
        })

    print(f"  Processed Entities: {entities}")

    # 4. Combine results
    parsed_data = {
        "original_message": message,
        "intent": {
            "name": intent,
            "confidence": intent_confidence
        },
        "entities": entities
    }

    return parsed_data

# --- 4. Main Execution Logic ---
if __name__ == "__main__":
    input_filename = "message.txt"
    output_filename = "result.json"

    # --- Read message from input file ---
    try:
        print(f"\nReading message from '{input_filename}'...")
        with open(input_filename, 'r', encoding='utf-8') as f:
            # Read the entire file content as the message
            message_to_process = f.read().strip()

        if not message_to_process:
            print(f"Error: Input file '{input_filename}' is empty or contains only whitespace.")
            exit() # Stop if the file is empty

        print("Message read successfully.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        print("Please create this file in the same directory as the script and add the message you want to parse.")
        exit() # Stop if the input file doesn't exist
    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_filename}': {e}")
        exit()

    # --- Process the message using the function ---
    try:
        final_result = parse_user_message(message_to_process)
    except Exception as e:
        print(f"An error occurred during message processing: {e}")
        # You might want more specific error handling depending on potential issues
        exit()

    # --- Write results to output JSON file ---
    try:
        print(f"\nWriting results to '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Use json.dump to write the dictionary to the file
            # ensure_ascii=False handles non-English characters correctly
            # indent=4 makes the JSON file readable
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        print("Results successfully written.")

    except Exception as e:
        print(f"An error occurred while writing to '{output_filename}': {e}")