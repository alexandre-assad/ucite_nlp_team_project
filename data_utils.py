"""
data_utils.py — Reading and writing CoNLL-U files

CoNLL-U is the standard format for Universal Dependencies treebanks.
Each sentence is a block of tab-separated lines, one per token,
separated by blank lines. Each line has 10 fields:
  ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC

We only keep the fields we need: ID, FORM, LEMMA, UPOS, HEAD, DEPREL.
We skip multi-word tokens (IDs like "1-2") and empty nodes (IDs like "1.1").
"""


def read_conllu(filepath):
    """
    Read a CoNLL-U file and return a list of sentences.
    
    Each sentence is a list of token dictionaries with keys:
      - id:     int, the token position (1-indexed)
      - form:   str, the word form
      - lemma:  str, the lemma
      - upos:   str, the universal POS tag
      - head:   int, the index of the head (0 means root)
      - deprel: str, the dependency relation label
    
    We also prepend a virtual ROOT token at position 0 for convenience,
    so that indexing matches the head pointers directly.
    """
    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # blank line means end of sentence
            if not line:
                if current_sentence:
                    # prepend the ROOT token at position 0
                    root_token = {
                        "id": 0,
                        "form": "ROOT",
                        "lemma": "ROOT",
                        "upos": "ROOT",
                        "head": -1,       # ROOT has no head
                        "deprel": "ROOT",
                    }
                    sentences.append([root_token] + current_sentence)
                    current_sentence = []
                continue

            # skip comment lines
            if line.startswith("#"):
                continue

            fields = line.split("\t")

            # skip multi-word tokens (e.g. "1-2") and empty nodes (e.g. "1.1")
            if "-" in fields[0] or "." in fields[0]:
                continue

            token = {
                "id": int(fields[0]),
                "form": fields[1],
                "lemma": fields[2],
                "upos": fields[3],
                "head": int(fields[6]),
                "deprel": fields[7],
            }
            current_sentence.append(token)

    # handle last sentence if file doesn't end with a blank line
    if current_sentence:
        root_token = {
            "id": 0,
            "form": "ROOT",
            "lemma": "ROOT",
            "upos": "ROOT",
            "head": -1,
            "deprel": "ROOT",
        }
        sentences.append([root_token] + current_sentence)

    return sentences


def write_conllu(filepath, sentences):
    """
    Write sentences with predicted heads and labels to a CoNLL-U file.
    
    Each sentence is a list of token dicts (including ROOT at index 0).
    We skip the ROOT token when writing, since it's virtual.
    The fields we don't predict (XPOS, FEATS, DEPS, MISC) are set to '_'.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token in sentence:
                # skip the virtual ROOT token
                if token["id"] == 0:
                    continue

                # build the 10-field CoNLL-U line
                fields = [
                    str(token["id"]),
                    token["form"],
                    token["lemma"],
                    token["upos"],
                    "_",                    # XPOS
                    "_",                    # FEATS
                    str(token["head"]),
                    token["deprel"],
                    "_",                    # DEPS
                    "_",                    # MISC
                ]
                f.write("\t".join(fields) + "\n")
            f.write("\n")  # blank line between sentences


def get_labels(sentences):
    """
    Collect all dependency relation labels found in a set of sentences.
    We exclude the ROOT label since it's not a real dependency relation.
    Returns a sorted list of unique labels.
    """
    labels = set()
    for sentence in sentences:
        for token in sentence:
            if token["deprel"] != "ROOT" and token["deprel"] != "_":
                labels.add(token["deprel"])
    return sorted(labels)
