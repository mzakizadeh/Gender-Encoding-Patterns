import argparse
import random

import numpy as np
import torch
from DataUtils import BiasInBios_extract_vectors_data
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    set_seed,
)

N_LABELS = 28


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights_type",
        type=str,
        required=True,
        choices=["basic", "finetuning", "random"],
        help="the model weights type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the feature extractor",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the feature extractor weights",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="experiment name to use in naming folders",
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default=None,
        choices=["raw", "scrubbed"],
        help="type of training data used to train the model",
    )
    parser.add_argument(
        "--training_balanced",
        type=str,
        default="original",
        choices=["subsampled", "oversampled", "original"],
        help="balancing of training data used to train the model",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="the random seed the model was trained on",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="raw",
        choices=["raw", "scrubbed", "name", "scrubbed_extra"],
        help="type of data to extract",
    )
    parser.add_argument(
        "--layer",
        "-z",
        type=int,
        default=-1,
        help="number of the layer which the experiment is conducting on",
    )
    parser.add_argument(
        "--max_length",
        default=128,
    )
    parser.add_argument(
        "--mappings_file",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()
    print("args: ", args)

    return args


def get_model(args):
    global model, folder

    feature_extractor = args.model_name if args.model_path is None else args.model_path
    if args.weights_type == "basic":
        config = AutoConfig.from_pretrained(
            feature_extractor,
            output_attentions=False,
            output_hidden_states=False,
            add_pooling_layer=False,
        )
        if args.adapter_path:
            model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=config)
            adapter_name = model.load_adapter(
                args.adapter_path,
                config="pfeiffer",
            )
            model.set_active_adapters(adapter_name)
        else:
            model = AutoModelForMaskedLM.from_pretrained(
                feature_extractor,
                trust_remote_code=True,
            )

    if args.weights_type == "random":
        config = AutoConfig.from_pretrained(
            feature_extractor,
            output_attentions=False,
            output_hidden_states=False,
            add_pooling_layer=False,
        )
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            # config=config,
        )
        model.apply(model._init_weights)

    elif args.weights_type == "finetuning":
        # TODO: check if we need this part
        pass

    print(model.config)

    folder = f"data/Vectorized_Data/{args.experiment_name}/{args.training_data}"
    folder += f"_{args.weights_type}_{args.training_balanced}"
    folder += f"_seed_{args.seed}"

    return model, folder


def __main__():
    args = parse_args()
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model, folder = get_model(args)
    model.to(device)

    tokens_path = f"data/Tokenized_Data/{args.experiment_name}/"
    tokens_path += (
        f"{args.training_data}_{args.model_name.replace('/', '-')}_{args.max_length}.pt"
    )

    BiasInBios_extract_vectors_data(
        args.data,
        model,
        tokens_path,
        args.experiment_name,
        folder,
        args.max_length,
        args.batch_size,
    )


if __name__ == "__main__":
    __main__()
