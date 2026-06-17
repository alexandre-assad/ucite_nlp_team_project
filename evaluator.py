"""
evaluator.py — Evaluation of Dependency Parsing

We use two standard metrics:

  1. UAS (Unlabeled Attachment Score): the percentage of tokens that are
     assigned the correct head, regardless of the label.

  2. LAS (Labeled Attachment Score): the percentage of tokens that are
     assigned both the correct head AND the correct label.

Both scores ignore the ROOT token (position 0) since it has no head.
"""


def evaluate(gold_sentences, predicted_sentences):
    """
    Compute UAS and LAS by comparing gold and predicted sentences.
    
    Parameters:
        gold_sentences:      list of sentences with gold heads/labels
        predicted_sentences:  list of sentences with predicted heads/labels
    
    Returns:
        (uas, las): both as percentages (0 to 100)
    """
    correct_head = 0     # number of tokens with correct head (for UAS)
    correct_both = 0     # number of tokens with correct head + label (for LAS)
    total = 0            # total number of tokens

    for gold_sent, pred_sent in zip(gold_sentences, predicted_sentences):
        # skip the ROOT token (index 0)
        for gold_tok, pred_tok in zip(gold_sent[1:], pred_sent[1:]):
            total += 1

            # check if the predicted head matches the gold head
            if pred_tok["head"] == gold_tok["head"]:
                correct_head += 1

                # check if the label also matches
                if pred_tok["deprel"] == gold_tok["deprel"]:
                    correct_both += 1

    # compute percentages
    uas = (correct_head / total * 100) if total > 0 else 0.0
    las = (correct_both / total * 100) if total > 0 else 0.0

    return uas, las


def print_evaluation(gold_sentences, predicted_sentences):
    """
    Evaluate and print the results in a readable format.
    """
    uas, las = evaluate(gold_sentences, predicted_sentences)
    total_tokens = sum(len(s) - 1 for s in gold_sentences)  # -1 for ROOT

    print(f"\n{'='*40}")
    print(f" Evaluation Results")
    print(f"{'='*40}")
    print(f" Sentences:  {len(gold_sentences)}")
    print(f" Tokens:     {total_tokens}")
    print(f" UAS:        {uas:.2f}%")
    print(f" LAS:        {las:.2f}%")
    print(f"{'='*40}\n")

    return uas, las
