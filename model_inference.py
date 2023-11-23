from transformers import BertTokenizerFast, BertForTokenClassification
import torch

model_dir = "model_save/"

tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForTokenClassification.from_pretrained(model_dir)

id2tag = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    predicted_labels = [id2tag[pred] for pred in predictions]

    token_label_pairs = list(zip(tokens, predicted_labels))

    return token_label_pairs

text = "We were just about to go up in the Kongur Shan now ."


token_label_pairs = predict(text)
for token, label in token_label_pairs:
    print(f'{token}: {label}')
print(predict(text))