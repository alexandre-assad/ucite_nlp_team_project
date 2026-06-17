"""
main.py — Entry Point for the Dependency Parser

This script ties all the pieces together. It provides a command-line
interface to:
  1. Train a parser on a French UD treebank
  2. Evaluate it on development and test sets
  3. Save and load the trained model

Usage:
  python main.py --train data/fr_gsd-ud-train.conllu \
                 --dev data/fr_gsd-ud-dev.conllu \
                 --test data/fr_gsd-ud-test.conllu \
                 --model model.pkl \
                 --epochs 10
"""

import argparse
import time

from component_registry import (
    resolve_decoder,
    resolve_feature_extractor,
    resolve_oracle,
    resolve_transition_system,
)
from data_utils import read_conllu, write_conllu, get_labels
from perceptron import AveragedPerceptron
from parser import train, parse_all
from evaluator import print_evaluation


def main():
    # ── Parse command-line arguments ──────────────────────────
    arg_parser = argparse.ArgumentParser(
        description="Transition-based dependency parser for French"
    )
    arg_parser.add_argument(
        "--train", type=str, default=None,
        help="Path to the training file (CoNLL-U format)"
    )
    arg_parser.add_argument(
        "--dev", type=str, default=None,
        help="Path to the development file (CoNLL-U format)"
    )
    arg_parser.add_argument(
        "--test", type=str, default=None,
        help="Path to the test file (CoNLL-U format)"
    )
    arg_parser.add_argument(
        "--model", type=str, default="model.pkl",
        help="Path to save/load the model (default: model.pkl)"
    )
    arg_parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    arg_parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed used for shuffling training sentences (default: 0)"
    )
    arg_parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write predicted trees (CoNLL-U format)"
    )
    arg_parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Optional directory for saving the best dev checkpoint during training"
    )
    arg_parser.add_argument(
        "--transition-system", type=str, default="arc-eager",
        choices=["arc-eager", "arc-standard"],
        help="Transition system to use (default: arc-eager)"
    )
    arg_parser.add_argument(
        "--oracle", type=str, default="static",
        choices=["static"],
        help="Oracle to use during training (default: static)"
    )
    arg_parser.add_argument(
        "--features", type=str, default="sparse",
        choices=["sparse"],
        help="Feature extractor to use (default: sparse)"
    )
    arg_parser.add_argument(
        "--decoder", type=str, default="greedy",
        choices=["greedy"],
        help="Decoder to use (default: greedy)"
    )
    args = arg_parser.parse_args()

    transition_system = resolve_transition_system(args.transition_system)
    oracle = resolve_oracle(args.oracle, args.transition_system)
    feature_extractor = resolve_feature_extractor(args.features)
    decoder = resolve_decoder(args.decoder)

    # ── Training mode ────────────────────────────────────────
    if args.train:
        print("=" * 60)
        print(" Transition-Based Dependency Parser")
        print(f" {args.transition_system} system + Averaged Perceptron")
        print("=" * 60)
        print()

        # load the training data
        print(f"Loading training data from {args.train}...")
        train_sentences = read_conllu(args.train)
        print(f"  → {len(train_sentences)} sentences loaded")

        # load the dev data (optional)
        dev_sentences = None
        if args.dev:
            print(f"Loading dev data from {args.dev}...")
            dev_sentences = read_conllu(args.dev)
            print(f"  → {len(dev_sentences)} sentences loaded")

        # collect all dependency labels from training data
        labels = get_labels(train_sentences)
        print(f"Found {len(labels)} dependency labels")
        print()

        # create the perceptron model
        model = AveragedPerceptron()

        # train!
        train(
            train_sentences,
            dev_sentences,
            model,
            labels,
            n_epochs=args.epochs,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            transition_system=transition_system,
            oracle=oracle,
            feature_extractor=feature_extractor,
        )

        # save the trained model
        model.save(args.model)

        # also save the labels (we need them for parsing)
        import pickle
        labels_path = args.model.replace(".pkl", "_labels.pkl")
        with open(labels_path, "wb") as f:
            pickle.dump(labels, f)
        print(f"Labels saved to {labels_path}")

    # ── Testing mode ─────────────────────────────────────────
    if args.test:
        print()
        print("=" * 60)
        print(" Evaluation on test set")
        print("=" * 60)

        # load the model if we didn't just train
        if not args.train:
            model = AveragedPerceptron()
            model.load(args.model)
            import pickle
            labels_path = args.model.replace(".pkl", "_labels.pkl")
            with open(labels_path, "rb") as f:
                labels = pickle.load(f)

        # load test data
        print(f"Loading test data from {args.test}...")
        test_sentences = read_conllu(args.test)
        print(f"  → {len(test_sentences)} sentences loaded")

        # parse all test sentences
        print("Parsing test sentences...")
        start = time.time()
        predicted = parse_all(
            test_sentences,
            model,
            labels,
            transition_system=transition_system,
            feature_extractor=feature_extractor,
            decoder=decoder,
        )
        elapsed = time.time() - start
        print(f"  Parsed {len(test_sentences)} sentences in {elapsed:.1f}s "
              f"({len(test_sentences)/elapsed:.0f} sent/s)")

        # evaluate
        print_evaluation(test_sentences, predicted)

        # optionally write predictions to a file
        if args.output:
            write_conllu(args.output, predicted)
            print(f"Predictions written to {args.output}")

    # ── If no train or test, show usage ──────────────────────
    if not args.train and not args.test:
        arg_parser.print_help()


if __name__ == "__main__":
    main()
