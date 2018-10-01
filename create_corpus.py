import re
import string
import sys
import warnings
from operator import itemgetter
from sys import version_info

import nltk
from unidecode import Cache, _warn_if_not_unicode


def main():
    LANGUAGES = {'de': 'german', 'en': 'english'}
    lang = LANGUAGES[sys.argv[1]]
    for line in sys.stdin:
        for sentence in process_line(line, language=lang):
            print(sentence)


def process_line(line, min_words=4, language='german'):
    sentences = []
    sents = nltk.sent_tokenize(line.strip(), language=language)
    for sentence in sents:
        sentence_processed = process_sentence(sentence, min_words)
        if sentence_processed:
            sentences.append(sentence_processed)

    return sentences


def process_sentence(sent, min_words=4):
    words = [normalize_word(word) for word in nltk.word_tokenize(sent, language='german')]
    if len(words) >= min_words:
        return ' '.join(w for w in words if w).strip()  # prevent multiple spaces
    return ''


def normalize_word(token):
    _token = unidecode_keep_umlauts(token)
    _token = remove_punctuation(_token)  # remove any special chars
    _token = replace_numeric(_token, by_single_digit=True)
    _token = '<num>' if _token == '#' else _token  # if token was a number, replace it with <unk> token
    return _token.strip().lower()


def remove_punctuation(text, punctiation_extended=string.punctuation + """"„“‚‘"""):
    return ''.join(c for c in text if c not in punctiation_extended)


def replace_numeric(text, numeric_pattern=re.compile('[0-9]+'), digit_pattern=re.compile('[0-9]'), repl='#',
                    by_single_digit=False):
    return re.sub(numeric_pattern, repl, text) if by_single_digit else re.sub(digit_pattern, repl, text)


def contains_numeric(text):
    return any(char.isdigit() for char in text)


def unidecode_keep_umlauts(text):
    # modified version from unidecode.unidecode_expect_ascii that does not replace umlauts
    _warn_if_not_unicode(text)
    try:
        bytestring = text.encode('ASCII')
    except UnicodeEncodeError:
        return _unidecode_keep_umlauts(text)
    if version_info[0] >= 3:
        return text
    else:
        return bytestring


def _unidecode_keep_umlauts(text):
    # modified version from unidecode._unidecode that keeps umlauts
    retval = []

    for char in text:
        codepoint = ord(char)

        # Basic ASCII, ä/Ä, ö/Ö, ü/Ü
        if codepoint < 0x80 or codepoint in [0xe4, 0xc4, 0xf6, 0xd6, 0xfc, 0xdc]:
            retval.append(str(char))
            continue

        if codepoint > 0xeffff:
            continue  # Characters in Private Use Area and above are ignored

        if 0xd800 <= codepoint <= 0xdfff:
            warnings.warn("Surrogate character %r will be ignored. "
                          "You might be using a narrow Python build." % (char,),
                          RuntimeWarning, 2)

        section = codepoint >> 8  # Chop off the last two hex digits
        position = codepoint % 256  # Last two hex digits

        try:
            table = Cache[section]
        except KeyError:
            try:
                mod = __import__('unidecode.x%03x' % (section), globals(), locals(), ['data'])
            except ImportError:
                Cache[section] = None
                continue  # No match: ignore this character and carry on.

            Cache[section] = table = mod.data

        if table and len(table) > position:
            retval.append(table[position])

    return ''.join(retval)


def check_lm(lm_path, vocab_path, sentence):
    import kenlm
    model = kenlm.LanguageModel(lm_path)
    print(f'loaded {model.order}-gram model from {lm_path}')
    print(f'sentence: {sentence}')
    print(f'score: {model.score(sentence)}')

    words = ['<s>'] + sentence.split() + ['</s>']
    for i, (prob, length, oov) in enumerate(model.full_scores(sentence)):
        two_gram = ' '.join(words[i + 2 - length:i + 2])
        print(f'{prob} {length}: {two_gram}')
        if oov:
            print(f'\t\"{words[i+1]}" is an OOV!')

    vocab = set(word for line in open(vocab_path) for word in line.strip().split())
    print(f'loaded vocab with {len(vocab)} unique words')
    print()
    word = input('Your turn now! Start a sentence by writing a word: (enter nothing to abort)\n')
    sentence = ''
    state_in, state_out = kenlm.State(), kenlm.State()
    total_score = 0.0
    model.BeginSentenceWrite(state_in)

    while word:
        sentence += ' ' + word
        sentence = sentence.strip()
        print(f'sentence: {sentence}')
        total_score += model.BaseScore(state_in, word, state_out)

        candidates = list((model.score(sentence + ' ' + next_word), next_word) for next_word in vocab)
        bad_words = sorted(candidates, key=itemgetter(0), reverse=False)
        top_words = sorted(candidates, key=itemgetter(0), reverse=True)
        worst_5 = bad_words[:5]
        print()
        print(f'least probable 5 next words:')
        for w, s in worst_5:
            print(f'\t{w}\t\t{s}')

        best_5 = top_words[:5]
        print()
        print(f'most probable 5 next words:')
        for w, s in best_5:
            print(f'\t{w}\t\t{s}')

        if '.' in word:
            print(f'score for sentence \"{sentence}\":\t {total_score}"')  # same as model.score(sentence)!
            sentence = ''
            state_in, state_out = kenlm.State(), kenlm.State()
            model.BeginSentenceWrite(state_in)
            total_score = 0.0
            print(f'Start a new sentence!')
        else:
            state_in, state_out = state_out, state_in

        word = input('Enter next word: ')

    print(f'That\'s all folks. Thanks for watching.')


if __name__ == '__main__':
    main()
