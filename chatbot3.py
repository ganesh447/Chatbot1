import pandas as pd
import faiss
import numpy as np
import torch
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import random

# Load FAQs
faqs = pd.read_csv("faq_cleaned.csv")
faqs["question"] = faqs["question"].str.lower()

# Create FAISS index for RAG
sent_model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = sent_model.encode(faqs["question"].tolist())
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(faq_embeddings, dtype=np.float32))

# Load RAG generator
generator = pipeline("text-generation", model="distilgpt2", framework="pt")

# Load BERT classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertForSequenceClassification.from_pretrained("bert_faq_classifier")
bert_tokenizer = BertTokenizer.from_pretrained("bert_faq_classifier")
bert_model.to(device)
bert_model.eval()
le = LabelEncoder()
le.classes_ = np.array(["Shipping", "Returns", "Account", "Support", "Other"])

# Mock product ID response
def get_order_details(product_id):
    return {
        "product_id": product_id,
        "status": "Processing",
        "delivery_date": "2025-04-18",
        "item_name": "Test Item",
        "last_updated": "2025-04-14"
    }

def classify_query(query, model, tokenizer, le):
    encodings = tokenizer(query.lower(), truncation=True, padding=True, max_length=64, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, dim=1)
    return le.inverse_transform(predicted.cpu().numpy())[0]

# Assign FAQ categories
faqs["category"] = ["Shipping" if any(w in q for w in ["track", "shipping", "delivery", "address", "package"]) else
                   "Returns" if any(w in q for w in ["return", "refund", "exchange", "damaged", "wrong item"]) else
                   "Account" if any(w in q for w in ["account", "password", "login", "update"]) else
                   "Support" if any(w in q for w in ["contact", "support", "chat", "issue"]) else
                   "Other" for q in faqs["question"]]

def get_relevant_faq(query, sent_model, index, faqs, category, k=1):
    cat_faqs = faqs[faqs["category"] == category]
    if cat_faqs.empty:
        cat_faqs = faqs
    cat_indices = cat_faqs.index.to_numpy()
    query_embedding = sent_model.encode([query.lower()])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    valid_idx = [i for i in indices[0] if i in cat_indices]
    if not valid_idx:
        valid_idx = indices[0][:1]
    return faqs.iloc[valid_idx[0]]["answer"]

def get_follow_up_question(category, faqs, history):
    # Avoid repeating recent questions
    recent_queries = [h["query"].lower() for h in history[-3:]] if history else []
    cat_faqs = faqs[faqs["category"] == category]
    if cat_faqs.empty:
        return None
    # Pick a random FAQ question not recently asked
    available = cat_faqs[~cat_faqs["question"].isin(recent_queries)]
    if available.empty:
        return None
    follow_up = available["question"].sample(1).iloc[0]
    # Format naturally
    return f"Would you like to know: {follow_up}"

def answer_query(query, history=None):
    if history is None:
        history = []

    # Check for product ID
    product_id_match = re.search(r"[A-Z]{3}\d{3}", query, re.IGNORECASE)
    if product_id_match:
        product_id = product_id_match.group(0).upper()
        order = get_order_details(product_id)
        answer = (f"Order {product_id} ({order['item_name']}): "
                  f"Status: {order['status']}, "
                  f"Expected delivery: {order['delivery_date']}, "
                  f"Last updated: {order['last_updated']}.")
        # No follow-up for product ID
        return answer, None

    # Check if responding to follow-up
    if query.lower() in ["yes", "sure", "okay", "yep"] and history:
        last_follow_up = history[-1].get("follow_up")
        if last_follow_up:
            # Extract original question (e.g., "Would you like to know: how long does shipping take?" -> "how long does shipping take?")
            follow_up_query = last_follow_up.replace("Would you like to know: ", "").rstrip("?")
            category = classify_query(follow_up_query, bert_model, bert_tokenizer, le)
            faq_answer = get_relevant_faq(follow_up_query, sent_model, index, faqs, category)
            prompt = f"{faq_answer}"
            answer = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"].strip()
            follow_up = get_follow_up_question(category, faqs, history)
            return answer, follow_up

    # New FAQ query
    category = classify_query(query, bert_model, bert_tokenizer, le)
    faq_answer = get_relevant_faq(query, sent_model, index, faqs, category)
    prompt = f"{faq_answer}"
    answer = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"].strip()
    follow_up = get_follow_up_question(category, faqs, history)
    return answer, follow_up

# Test
if __name__ == "__main__":
    history = []
    x = input("Enter your query ")
    test_queries = [x]
    for query in test_queries:
        answer, follow_up = answer_query(query, history)
        print(f"Answer: {answer}")
        if follow_up:
            print(f"Follow-up: {follow_up}")
        print()
        history.append({"query": query, "answer": answer, "follow_up": follow_up})