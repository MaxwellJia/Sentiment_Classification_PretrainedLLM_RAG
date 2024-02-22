import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import json
import os


# load the training data
train_data = json.load(open(os.path.join("data", "genre_train.json"), "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid'] # these are the ids of the books which each training example came from


# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open(os.path.join("data", "genre_test.json"), "r"))
Xt = test_data['X']


# create a distilbert-base-uncased model and assign some parameters
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=300):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return input_ids, attention_mask, label
        return input_ids, attention_mask


# Model Definition
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Datasets
train_dataset = TextDataset(X, Y)
test_dataset = TextDataset(Xt)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training Setup
optimizer = AdamW(model.parameters(), lr=3e-5)
epochs = 20
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training Loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)

        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}, Loss: {avg_train_loss}")

# Prediction
model.eval()
predictions = []
for batch in test_loader:
    b_input_ids, b_input_mask = batch
    b_input_ids = b_input_ids.to(device)
    b_input_mask = b_input_mask.to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    predictions.extend(preds)

# Save Predictions
with open("out.csv", "w") as fout:
    fout.write("Id,Predicted\n")
    for i, pred in enumerate(predictions):
        fout.write(f"{i},{pred}\n")