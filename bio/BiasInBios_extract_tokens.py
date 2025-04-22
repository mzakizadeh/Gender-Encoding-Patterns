import argparse

from DataUtils import BiasInBios_extract_tokens_data
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        type=str,
        help="type of training data used to train the model",
        choices=["raw", "scrubbed"],
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_version for tokenizer",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to `BIOS.pkl`",
        default="BIOS.pkl",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="maximum length of tokens",
    )
    parser.add_argument(
        "--mappings_file",
        type=str,
    )

    args = parser.parse_args()
    print("args: ", args)

    return args


def __main__():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    BiasInBios_extract_tokens_data(
        args.type,
        tokenizer,
        args.model_name,
        args.data_path,
        args.max_length,
        args.experiment_name,
        args.mappings_file,
    )


if __name__ == "__main__":
    __main__()
