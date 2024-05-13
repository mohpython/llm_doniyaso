#!/usr/bin/env python
import click
from nltk.tokenize.punkt import PunktSentenceTokenizer
from brat_parser import get_entities_relations_attributes_groups


@click.command()
@click.argument('abstractfile')
@click.argument('labelfile')
def main(abstractfile, labelfile):
    entities, relations, attributes, groups = get_entities_relations_attributes_groups(labelfile)

    targets = [e for e in entities.values() if e.type == 'Target']
    print(targets)

    with open(abstractfile, 'r') as f:
        contents = f.read()

    # Get the (start, end) index of each sentence in the abstract
    sentence_splitter = PunktSentenceTokenizer()
    sentence_spans = sentence_splitter.span_tokenize(contents)

    # Print any sentence containing a target name
    for start, end in sentence_spans:
        if any(start <= t.span[0][0] and t.span[0][1] <= end for t in targets):
            print(contents[start:end])


if __name__ == '__main__':
    main()
