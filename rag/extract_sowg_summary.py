#!/usr/bin/env python
"""
Extract SOWG Documentarian Summary from analyst notebook documents
"""
import os
import click
from glob import glob
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag, NavigableString


def no_summary_header(soup):
    return None


def summary_header(soup, header):

    txt = ''

    for d in header.parent.children:
        if isinstance(d, Tag) and d.string is not None:
            if d.name in ('h1', 'h2'):
                end = '\n'
            else:
                end = ''
            txt += f'{d.string}{end}'
        elif isinstance(d, NavigableString):
            txt += d

    return txt


def extract_sowg_summary(doc_html):

    with open(doc_html, 'rb') as fp:
        soup = BeautifulSoup(fp, "html.parser")

    target = soup.find('h2', string='Summary')

    if target is None:
        return no_summary_header(soup)
    else:
        return summary_header(soup, target)


def get_output(outputdir, inputfile):
    return os.path.join(
        outputdir,
        os.path.splitext(os.path.basename(inputfile))[0] + '.txt'
    )


@click.command()
@click.argument('inputdir')
@click.argument('outputdir')
def main(inputdir, outputdir):

    html_files = sorted(glob(os.path.join(inputdir, '*_0049.htm')))

    for hf in tqdm(html_files, 'Extracting'):
        summary = extract_sowg_summary(hf)
        if summary is None: continue

        ofile = get_output(outputdir, hf)
        with open(ofile, 'w') as f:
            f.write(summary)


if __name__ == '__main__':
    main()
