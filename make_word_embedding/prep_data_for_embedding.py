from data.data_utils import read_tsv_lists
import string
import gzip


def prep_data_embedding(txt):
    # remove puncts
    PUNCTS = string.punctuation.replace("'", "")
    # text is case folded, stripped and puncts are removed 
    txt = txt.lower().strip().translate(str.maketrans(PUNCTS, ' '*len(PUNCTS))) + " \n"
    return txt

# read data
targets, premise, hypothesis, _, _ = read_tsv_lists("/fp/projects01/ec30/IN5550/mnli_train.tsv.gz")
# combine the lists of hypothesis and premise
txt = hypothesis + premise
# prep data
txt_prepped = [prep_data_embedding(p) for p in txt]

print(f"# of docs = {len(txt_prepped)}")
print(f"# of tokens = {sum([len(p.split()) for p in txt_prepped])}")

# write prepped data to gz file
with gzip.open('/fp/projects01/ec30/magnuspn/big_files/prepped_corpus.gz', 'wb') as f:
    for pair in txt_prepped:
        f.write(pair.encode())
