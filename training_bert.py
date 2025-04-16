import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

# Load dataset
data = pd.read_csv("classifier_data.csv")
queries = data["query"].str.lower().tolist()
labels = data["label"].tolist()

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
num_labels = len(le.classes_)  # 5

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(queries, truncation=True, padding=True, max_length=64, return_tensors="pt")
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = torch.tensor(encoded_labels)

# Dataset
class FAQDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

dataset = FAQDataset(input_ids, attention_mask, labels)

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
epochs = 3

# Train
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Validation Accuracy: {accuracy:.4f}")

# Save
model.save_pretrained("bert_faq_classifier")
tokenizer.save_pretrained("bert_faq_classifier")