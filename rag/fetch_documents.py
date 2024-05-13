#!/usr/bin/env python
import os
import re
import click
import requests
from tqdm import tqdm

BASE = 'https://an.rsl.wustl.edu'
URL = BASE + f'/msl/mslbrowser/document.aspx?d={{docid}}'
DOC_URL_RE = '/msl/mslbrowser/docs/([0-9\-]{5,11})/([0-9]{4})/([0-9\-]{5,11})_([0-9]{4}).htm'


def get_doc_url(html):

    # Search for the URL pattern in the HTML
    match = re.search(DOC_URL_RE, html)
    if match is None:
        return None, None, None

    # Get the full URL
    url = BASE + match.group(0)

    # Get the first part of the document id
    id1 = match.group(3)
    if id1 != match.group(1):
        print(f'Warning: Unexpected URL format {url}')

    # Get the second part of the document id
    id2 = match.group(4)
    if id2 != match.group(2):
        print(f'Warning: Unexpected URL format {url}')

    return url, id1, id2


@click.command()
@click.argument('outputdir')
@click.option('-s', '--startdoc', default=1)
@click.option('-e', '--enddoc', default=5593)
def main(outputdir, startdoc, enddoc):

    for d in tqdm(range(startdoc, enddoc + 1), 'Fetching Documents'):
        url = URL.format(docid=d)

        # Get response and raise any errors
        response = requests.get(url)
        if response.status_code != 200:
            print(f'Warning: received status code {response.status_code} for {url}')
            continue

        # Get document URL from response
        doc_url, id1, id2 = get_doc_url(response.text)
        if doc_url is None:
            print(f'No document URL found for {url}')
            continue

        docfile = os.path.join(outputdir, f'{id1}_{id2}.htm')

        # Skip if already downloaded
        if os.path.exists(docfile): continue

        # Fetch document
        doc = requests.get(doc_url)
        if doc.status_code != 200:
            print(f'Warning: received status code {doc.status_code} for {url}')
            continue

        # Save document
        with open(docfile, 'wb') as f:
            f.write(doc.content)


if __name__ == '__main__':
    main()
