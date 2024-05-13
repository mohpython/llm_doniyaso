#!/usr/bin/env python
import os
import re
import click # type:
from nltk.tokenize.punkt import PunktSentenceTokenizer
from brat_parser import get_entities_relations_attributes_groups
from seaborn import get_dataset_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification


def tokenize_sentences(contents):
    sentence_splitter = PunktSentenceTokenizer()
    sentences = [{'text': contents[start:end], 'start': start, 'end': end} for start, end in sentence_splitter.span_tokenize(contents)]
    return sentences


def find_sentences_with_targets(sentences, targets):
    sentences_with_targets = []
    for sentence in sentences:
        has_target = any(target.span[0][0] >= sentence['start'] and target.span[0][1] <= sentence['end'] for target in targets)
        sentences_with_targets.append({'text': sentence['text'], 'has_target': has_target})
    return sentences_with_targets


@click.command()
@click.argument('abstractfile')
@click.argument('labelfile')
def main(abstractfile, labelfile):
    entities, _, _, _ = get_entities_relations_attributes_groups(labelfile)
    targets = [e for e in entities.values() if e.type == 'Target']
    
    with open(abstractfile, 'r') as f:
        contents = f.read()

    sentences = tokenize_sentences(contents)
    sentences_with_targets = find_sentences_with_targets(sentences, targets)

    # Préparation des données pour le modèle
    sentences_text = [s['text'] for s in sentences_with_targets]
    labels = [1 if s['has_target'] else 0 for s in sentences_with_targets]

    train_texts, test_texts, train_labels, test_labels = train_test_split(sentences_text, labels, test_size=0.2, random_state=42)

    # Tokenization avec BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Entraînement du modèle
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_loader = torch.utils.data.DataLoader(model, batch_size=8, shuffle=True)

    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Évaluation du modèle
    model.eval()
    predictions = []
    test_loader = torch.utils.data.DataLoader(get_dataset_names, batch_size=8, shuffle=False)
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predictions.extend(torch.argmax(probabilities, dim=1).tolist())

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    print("\nExpérience 1 - Modèle de classification binaire avec BERT :")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == '__main__':
    main()
