# wiki-lm
Script to train a n-gram Language Model (LM) of any order on any language with Wikipedia articles. See [my blog post](https://tiefenauer.github.io/blog/wiki-n-gram-lm/) for details.

## Usage:
The script is located at [create_lm.sh](./create_lm.sh). Type `./create_lm.sh -h` to display the help.

### Example usage
```bash
# create a 4-gram LM for German using the 400k most frequent words and probing as data structure. Artifacts will be removed after estimation. 
./create_lm.sh -l de -o 4 -t 400000 -d probing -r
```

## Dependencies
The script assumes the following dependencies are available on your system:

- [KenLM](https://github.com/kpu/kenlm): KenLM is used to create the LM and assumes that `lmplz` and `build_binary` are on `$PATH`. See [the KenLM docs](https://kheafield.com/code/kenlm/) for more information about how to build those binaries.
- [Pipeline Viewer](http://www.ivarch.com/programs/pv.shtml) to show a progress bar

## Corpus creation
The script will perform all steps necessary to create a corpus from the Wikipedia articles that can be used for estimation. Some of the logic used for creating the corpus is contained in [create_corpus.py](./create_corpus.py). The main steps are the following:

1. Download the dump from Wikipedia
2. Remove Wiki markup and extract raw text from articles. This step uses the [Wikipedia Extractor](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor) which was implemented by me and was simply copied from [the GitHub Repo](https://github.com/attardi/wikiextractor).
3. Tokenize the text into sentences using [NLTK](https://nltk.org). All text is normalized. The result is used to write a compressed corpus file with one sentence per line, words separated by a single whitespace (as expected by KenLM). Normalization is done as follows:
   - try to convert any non-ASCII characters to their ASCII equivalents using [unidecode](https://pypi.org/project/Unidecode/), if this is possible. This is neccessary to reduce possible spelling errors with accentuated characters and get rid of ambigous spelling variants (like e.g. the `ß` used in German that is sometimes also written as `ss`). Umlauts (`äöü`) will not be replaced.
   - punctuation is removed (this includes any punctuation used to mark the end of a sentence)
   - replace any purely numeric word-tokens within each sentence (year numbers etc.) by the `<num>` token. This is done because such tokens usually do not carry any semantic meaning and can be replaced by any other number. Word-tokens containing a digit (e.g. _WW2_) will be processed by replacing the digit by `#` (e.g. _WW#_).
   - remove any whitespace at the beginning and end of each sentence as well as multiple whitespaces between 
   - make everything lowercase
4. Estimate the probability of n-grams using `lmplz` and create an [ARPA](https://cmusphinx.github.io/wiki/arpaformat/) file containing the estimations  
5. Create a binary KenLM model from the ARPA file using `build_binary` that can be loaded by the `kenlm` python module and used for fast estimation.

(c) Daniel Tiefenauer