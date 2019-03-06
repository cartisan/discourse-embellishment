import nltk
from pprint import pprint
from collections import Counter, defaultdict


STORY_CORPUS = "story_corpus.orig.txt"
STORY_SEP = "=="
SYSTEM_SEP = "===="

DATASET = "wikilarge"


def parse_data(file_loc):
    print("reading in: " + file_loc)
    with open(file_loc) as f:
        lines = f.readlines()

    # remove story and system separators
    return [l for l in lines if not
            (l.startswith(STORY_SEP) or l.startswith(SYSTEM_SEP))]


def tokenize(lines):
    print("Tokenizing " + str(len(lines)) + " lines...")
    tokenized_lines = []
    for l in lines:
        tokenized_lines.append(nltk.word_tokenize(l))
    return [" ".join(fix_q_marks(l)) for l in tokenized_lines]


def save_data(lines, filename):
    with open(filename, "w") as f:
        for line in lines:
            f.write(line + "\n")


def fix_q_marks(line):
    if DATASET == "docaligned":
        return line
    if DATASET == "wikilarge":
        return [l.replace("``", "''").replace("''", "''") for l in line]


def iterate_pairs(l):
    pairs = list(zip(l, l[1:]))[::2]
    if (len(l) % 2 == 1):
        # if l has uneven number of elements, zip ignores the (unpaired)
        # last one
        pairs.append((l[-1], ""))
    return pairs


def split_in_sentences(lines):
    text = " ".join(lines)
    return nltk.tokenize.sent_tokenize(text)


def create_sent_pairs(lines):
    tup_lines = []
    for sen1, sen2 in iterate_pairs(lines):
        tup_lines.append(sen1 + " " + sen2)
    return tup_lines


def replace_ne(lines):
    new_lines = []
    for sent in lines:
        new_sent = []
        tag_counter = Counter()
        tag_dict = {}
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                entity = chunk[0][0].lower()
                ner_tag = chunk.label()
                if ner_tag == "GPE":
                    ner_tag = "LOCATION"

                if entity+ner_tag not in tag_dict.keys():
                    tag_counter[ner_tag] += 1
                    tag_dict[entity+ner_tag] = tag_counter[ner_tag]
                new_sent.append(ner_tag + "@" + str(tag_dict[entity+ner_tag]))
                # new_sent.append(ner_tag + "@" + str(tag_counter[ner_tag]))
            else:
                # new_sent.append(' '.join(c[0] for c in chunk))
                new_sent.append(chunk[0])
                assert len(chunk) == 2
        new_lines.append(" ".join(new_sent))

    return new_lines


DATASET = "docaligend"

lines = parse_data(STORY_CORPUS)
if DATASET == "docaligned":
    lines = tokenize(lines)

elif DATASET == "wikilarge":
    lines = parse_data(STORY_CORPUS)
    lines = split_in_sentences(lines)
    lines = tokenize(lines)
    lines = create_sent_pairs(lines)
    lines = replace_ne(lines)

save_data(lines, "story_corpus.txt")

# pprint(lines)
# import pdb;pdb.set_trace()
