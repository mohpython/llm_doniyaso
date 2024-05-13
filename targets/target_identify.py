#!/usr/bin/env python
import os
import re
import click
import requests
from nltk.tokenize.punkt import PunktSentenceTokenizer
from brat_parser import get_entities_relations_attributes_groups
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score


def tokenize_sentences(contents):
    sentence_splitter = PunktSentenceTokenizer()
    sentences = [{'text': contents[start:end], 'start': start, 'end': end} for start, end in sentence_splitter.span_tokenize(contents)]
    return sentences


def find_sentences_with_targets(sentences, targets):
    sentences_with_targets = []
    for sentence in sentences:
        s_targets = [
            target.text for target in targets
            if target.span[0][0] >= sentence['start'] and target.span[0][1] <= sentence['end']
        ]
        has_target = (len(s_targets) > 0)
        sentences_with_targets.append({'text': sentence['text'], 'has_target': has_target, 'targets': s_targets})
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

    # Utiliser un pipeline de Hugging Face pour l'identification des entités nommées
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    # Analyser les phrases pour identifier les entités nommées (cibles géologiques)
    predicted_targets = []
    for text in tqdm(sentences, 'Extracting entities'):
        entities = ner_pipeline(text['text'])
        targets = [
            entity['word'] for entity in entities
            if entity['entity'].upper() in ('B-GEO', 'I-LOC', 'I-PER')
        ]
        predicted_targets.append(targets)

    # Comparer les prédictions avec les cibles réelles et calculer la précision
    correct_predictions = [
        set(predicted) == set(actual['targets'])
        for predicted, actual in zip(predicted_targets, sentences_with_targets)
    ]
    accuracy_llm = np.mean(correct_predictions)

    print("\nExpérience 2 - Identification des cibles par LLM :")
    print(f"Accuracy: {accuracy_llm}")


if __name__ == '__main__':
    main()