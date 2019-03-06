from collections import defaultdict
from pprint import pprint

NORM = "normal.txt"
SIMP = "simple.txt"

TRAIN = 0.85
VAL = 0.1
TEST = 0.05


def parse_data(file_loc):
    print "reading in: " + file_loc
    with open(file_loc) as f:
        text = f.readlines()

    print " parsing it into dict form..."
    result_dic = defaultdict(lambda: defaultdict(list))
    for line in text:
        doc, para, text = line.split("\t")
        result_dic[doc][int(para)].append(text[:-1])  # remove newline at line end

    print " number of documents found:", len(result_dic.keys())  # should be 59775 for simple

    return result_dic


def determine_split(simple_dic):
    print "computing sentence number per document..."
    doc_sent_num_map = defaultdict(int)
    sen_count = 0.0
    for doc in simple_dic.keys():
        for para in simple_dic[doc]:
            doc_sent_num_map[doc] += len(simple_dic[doc][para])
            sen_count += len(simple_dic[doc][para])

    # print doc_sent_num_map["April"]  # should be 10
    # pprint(sorted(doc_sent_num_map.keys())[:50])
    print " overall sentence number: ", sen_count

    print "splitting dataset (train/val/test):", TRAIN, VAL, TEST
    counter = 0
    train = []
    val = []
    test = []
    for doc, sen_num in doc_sent_num_map.items():
        counter += sen_num
        percentile = counter / sen_count

        if percentile < TRAIN:
            train.append(doc)
        elif percentile < (TRAIN + VAL):
            val.append(doc)
        else:
            test.append(doc)

    print " Resulting split:"
    print "   train:", compute_split_size(train, doc_sent_num_map) / sen_count
    print "   val:", compute_split_size(val, doc_sent_num_map) / sen_count
    print "   test:", compute_split_size(test, doc_sent_num_map) / sen_count

    print "Saving split keys..."
    save_split(train, "train_docs.txt")
    save_split(val, "val_docs.txt")
    save_split(test, "test_docs.txt")

    return train, val, test


def load_split(train_name, val_name, test_name):
    results = []
    for f_name in [train_name, val_name, test_name]:
        with open(f_name, "r") as f:
            text = f.readlines()
        results.append([topic[:-1] for topic in text])

    return results[0], results[1], results[2]


def create_split_per_paragraph(docs, simple_dic, normal_dic, split_name):
    print "Creating actual split files for:", split_name, "..."
    with open("docaligned.normal." + split_name, "w") as f_norm:
        with open("docaligned.simple." + split_name, "w") as f_simp:
            for doc in docs:
                for para in simple_dic[doc]:
                    line_s = " ".join(simple_dic[doc][para])
                    line_n = " ".join(normal_dic[doc][para])
                    if line_s.strip() and line_n.strip():
                        # include only if both datasets have this line
                        f_simp.write(line_s + "\n")
                        f_norm.write(line_n + "\n")


def compute_split_size(split, doc_sent_num_map):
    count = 0
    for doc in split:
        count += doc_sent_num_map[doc]
    return count


def save_split(docs, filename):
    with open(filename, "w") as f:
        for doc in docs:
            f.write(doc+"\n")


# #### parse the dataset
simple_dic = parse_data(SIMP)
normal_dic = parse_data(NORM)

# #### idetermine how to split dataset into train/eval/test sets, according to line counts
# train, val, test = determine_split(simple_dic)  # to re-generate split, uncomment this
train, val, test = load_split("train_docs.txt", "val_docs.txt", "test_docs.txt")


# ### create actual splits
create_split_per_paragraph(train, simple_dic, normal_dic, "train")
create_split_per_paragraph(val, simple_dic, normal_dic, "val")
create_split_per_paragraph(test, simple_dic, normal_dic, "test")
