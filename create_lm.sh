#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h] [-o <int>] [-l {'en'|'de'|'fr'|'it'|...}] [-d {'probing'|'trie'}] [-t <string> ] [-w <int>] [-r remove_artifacts] -- Create n-gram Language Model on ~2.2M Wikipedia articles using KenLM.
where:
    -h  show this help text
    -o  set the order of the model, i.e. the n in n-gram (default: 4)
    -l  ISO 639-1 code of the language to train on (default: de)
    -d  data structure to use (use 'trie' or 'probing'). See https://kheafield.com/code/kenlm/structures/ for details. (default: trie)
    -t  target directory to write to
    -w  number of words in vocabulary to keep (default: 500,000)
    -r  remove intermediate artifacts after training. Only set this flag if you really don't want to train another model because creating intermediate artifacts can take a long time. (default: false)

EXAMPLE USAGE: create a 4-gram model for German using the 400k most frequent words from the Wikipedia articles, using probing as data structure and removing everything but the trained model afterwards:

./create_lm.sh -l de -o 4 -t 400000 -r

Make sure the target directory specified by -t has enough free space (around 20-30G). KenLM binaries (lmplz and build_binary) need to be on the path. See https://kheafield.com/code/kenlm/ on how to build those.

The following intermediate artifacts are created and may be removed after training by setting the -r flag:
- {target_dir}/tmp/[language]wiki-latest-pages-articles.xml.bz2: Downloaded wikipedia dump
- {target_dir}/tmp/[language]_clean: directory containing preprocessed Wikipedia articles
- {target_dir}/tmp/wiki_[language].txt.bz2: compressed file containing the Wikipedia corpus used to train the LM (raw text contents of the Wikipedia articles one sentence per line)
- {target_dir}/tmp/wiki_[language].counts: file containing the full vocabulary of the corpus and the number of occurrences of each word (sorted descending by number of occurrences)
- {target_dir}/tmp/wiki_[language].vocab: file containing the most frequent words of the corpus used for training (as defined by the -t argument) in the format expected by KenLM (words separated by spaces)
- {target_dir}/tmp/wiki_[language].arpa: ARPA file used to create the KenLM binary model

The following result files are created and will not be removed:
- {target_dir}/wiki_[language].klm: final KenLM n-gram LM in binary format.
"

# Defaults
order=4
language='de'
data_structure=trie
top_words=500000
target_dir='./'
remove_artifacts=false

while getopts ':hs:' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    o) order=$OPTARG
       ;;
    l) language=$OPTARG
       ;;
    d) data_structure=$OPTARG
       ;;
    w) top_words=$OPTARG
       ;;
    t) target_dir=$OPTARG
       ;;
    r) remove_artifacts=true
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

# #################################
# Paths and filenames
# #################################
corpus_name="wiki_${language}"
lm_basename="${corpus_name}_${order}_gram"
tmp_dir="${target_dir}/tmp"  # directory for intermediate artifacts

cleaned_dir="${tmp_dir}/${corpus_name}_clean" # directory for WikiExtractor
corpus_file="${tmp_dir}/${corpus_name}.txt" # uncompressed corpus
lm_counts="${tmp_dir}/${corpus_name}.counts" # corpus vocabulary with counts (all words)
lm_vocab="${tmp_dir}/${corpus_name}.vocab" # corpus vocabulary used for training (most frequent words)
lm_arpa="${tmp_dir}/${lm_basename}.arpa" # ARPA file

lm_binary="${target_dir}/${lm_basename}.klm" # KenLM binary file (this is the result of the script)

# create target directories if the don't exist yet
mkdir -p $target_dir
mkdir -p $tmp_dir
# #################################

echo "creating $order-gram model from Wikipedia dump"
echo "time indications are based upon personal experience when training on my personal laptop (i7, 4 cores, 8GB RAM, SSD)"

# recreate vocabulary
[ -f $lm_vocab ]
recreate_vocab=$?

# #################################
# STEP 1: Download the Wikipedia dump in the given language if necessary
# For a some statistics of Wikipedias see https://meta.wikimedia.org/wiki/List_of_Wikipedias
# #################################
download_url="http://download.wikimedia.org/${language}wiki/latest/${language}wiki-latest-pages-articles.xml.bz2"
target_file=${tmp_dir}/$(basename ${download_url})  # get corpus file name from url and corpus name
if [ ! -f ${target_file} ]; then
    echo "downloading corpus ${corpus_name} from ${download_url} and saving in ${target_file}"
    echo "This can take up to an hour (Wiki servers are slow). Have lunch or something..."
    wget -O ${target_file} ${download_url}
fi

# #################################
# STEP 2: Create corpus from dump if necessary
# Use WikExtractor (see https://github.com/attardi/wikiextractor for details)
# #################################
if [ ! -f "${corpus_file}" ] ; then
    if [ ! -d $cleaned_dir ] ; then
        echo "Extracting/cleaning text from Wikipedia data base dump at ${target_file} using WikiExtractor."
        echo "Cleaned articles are saved to ${cleaned_dir}"
        echo "This will take 2-3 hours. Have a walk or something..."
        mkdir -p ${cleaned_dir}
        python3 ./WikiExtractor.py -c -b 25M -o ${cleaned_dir} ${target_file}
    fi
    echo "Uncompressing and preprocessing cleaned articles from $cleaned_dir"
    echo "All articles will be written to $corpus_file (1 sentence per line, without dot at the end)."
    echo "All XML tags will be removed. Numeric word tokens will be replaced by the <num> token."
    echo "Non-ASCII characters will be replaced with their closest ASCII equivalent (if possible), but umlauts will be preserved!"
    echo "This will take some time (~4h). Go to sleep or something..."
    result=$(find $cleaned_dir -name '*bz2' -exec bzcat {} \+ \
            | pv \
            | tee >(    sed 's/<[^>]*>//g' \
                      | sed 's|["'\''„“‚‘]||g' \
                      | python3 ./create_corpus.py ${language} > ${corpus_file} \
                   ) \
            | grep -e "<doc" \
            | wc -l)
    echo "Processed ${result} articles and saved raw text in $corpus_file"

    echo "Processed $(cat ${corpus_file} | wc -l) sentences"
    echo "Processed $(cat ${corpus_file} | wc -w) words"
    echo "Processed $(cat ${corpus_file} | xargs -n1 | sort | uniq -c) unique words"

    echo "compressing $corpus_file. File size before:"
    du -h ${corpus_file}
    bzip2 ${corpus_file}
    echo "done! Compressed file size:"
    du -h ${corpus_file}.bz2

    # vocabulary must be recreated because corpus might have changed
    recreate_vocab = 1
fi

if [ ${recreate_vocab} = 1 ] ; then
    echo "(re-)creating vocabulary of $corpus_file and saving it in $lm_vocab. "
    echo "This usually takes around half an hour. Get a coffee or something..."

    echo "counting word occurrences..."
    cat ${corpus_file} |
        pv -s $(stat --printf="%s" ${corpus_file}) | # show progress bar
        tr '[:space:]' '[\n*]' | # replace space with newline (one word per line)
        grep -v "^\s*$" | # remove empty lines
        grep -v '#' | # remove words with numbers
        awk 'length($0)>1' | # remove words with length 1
        sort | uniq -c | sort -bnr > ${lm_counts} # sort alphanumeric, count unique words, then sort result

    echo "...done! writing $top_words top words to vocabulary"
    cat ${lm_counts} |
        tr -d '[:digit:] ' | # remove counts from lines
        head -${top_words} | # keep $top_words words
        tr '\n' ' ' > ${lm_vocab} # replace newline with spaces (expected input format for KenLM)

    total_sum=$(cat ${lm_counts} |
            tr -d ' [:alpha:]äöü<>\177' | # remove non-numeric characters (everything but the counts)
            paste -sd+ | # concatenate with '+'
            bc) # sum up
    top_sum=$(cat ${lm_counts} |
            head -${top_words} | # limit to first $top_words entries here
            tr -d ' [:alpha:]äöü<>\177' | # same as above
            paste -sd+ | bc) # same as above
    fraction=$(echo "scale=2 ; 100 * $top_sum / $total_sum" | bc)
    echo "Top $top_words words make up $fraction% of words"
fi


if [ ! -f $lm_arpa ]; then
    echo "Training $order-gram KenLM model with data from $corpus_file.bz2 and saving ARPA file to $lm_arpa"
    echo "This can take several hours, depending on the order of the model"
    lmplz -o ${order} -T ${tmp_dir} -S 40%  --limit_vocab_file ${lm_vocab} <${corpus_file}.bz2
fi

if [ ! -f $lm_binary ]; then
    echo "Building binary file from $lm_arpa and saving to $lm_binary"
    echo "This should usually not take too much time even for high-order models"
    build_binary ${data_structure} ${lm_arpa} ${lm_binary}
fi

if ${remove_artifacts}; then
    echo "removing intermediate artifacts in ${tmp_dir}"
    rm -rf ${tmp_dir}
fi
