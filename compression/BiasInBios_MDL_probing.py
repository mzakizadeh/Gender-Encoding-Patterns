import torch
from MDLProbingUtils import general_MDL_args, run_MDL_probing

import wandb


def parse_args():
    parser = general_MDL_args()
    parser.add_argument(
        "--weights_type",
        required=True,
        type=str,
        help="the model type",
        choices=["basic", "finetuning", "random"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model to use as a feature extractor",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="experiment name to use in naming folders",
    )
    parser.add_argument(
        "--training_data",
        required=False,
        type=str,
        help="the data type the model was trained on",
        choices=["raw", "scrubbed"],
        default=None,
    )
    parser.add_argument(
        "--training_balanced",
        type=str,
        help="balancing of the training data",
        choices=["subsampled", "oversampled", "original"],
        default="original",
    )
    parser.add_argument(
        "--type",
        type=str,
        help="the type of vectors to probe",
        choices=["raw", "scrubbed"],
    )
    parser.add_argument(
        "--testing_balanced",
        type=str,
        help="balancing of the testing data",
        choices=["subsampled", "oversampled", "original"],
        default="original",
    )
    parser.add_argument(
        "--layer",
        "-z",
        type=int,
        help="number of the layer which the experiment is conducting on",
        default=-1,
    )
    parser.add_argument(
        "--max_length",
        default=128,
    )

    args = parser.parse_args()
    print("args: ", args)

    return args


def load_bias_in_bios_vectors(args):
    folder = (
        f"data/Vectorized_Data/{args.experiment_name}/{args.training_data}"
        f"_{args.weights_type}_{args.training_balanced}_seed_{args.seed}"
    )

    layer = "last" if args.layer == -1 else f"{args.layer}"

    vectors = torch.load(
        folder
        + f"/vectors_{args.type}_{args.experiment_name}_{args.max_length}_layer_{layer}.pt"
    )

    train = {}
    for i in range(int(len((vectors["X"]) - 1) * 0.8)):
        train[i] = {
            "X": vectors["X"][i],
            "y": vectors["z"][i],
            "z": vectors["z"][i],
        }

    valid = {}
    index = 0
    for i in range(int(len((vectors["X"]) - 1) * 0.8), len(vectors["X"])):
        valid[index] = {
            "X": vectors["X"][i],
            "y": vectors["z"][i],
            "z": vectors["z"][i],
        }
        index += 1

    test = valid
    return train, valid, test


def init_wandb(args):
    wandb.init(
        project="BiosBias Compressions",
        name=f"{args.experiment_name}_type_{args.weights_type}_seed_{args.seed}_layer_{args.layer}",
        config={
            "model_name": args.model_name,
            "experiment_name": args.experiment_name,
            "seed": args.seed,
            "model_seed": args.model_seed,
            "training data": args.training_data,
            "training balancing": args.training_balanced,
            "testing balancing": args.testing_balanced,
            "type": args.type,
            "weights_type": args.weights_type,
            "layer": args.layer,
        },
    )


def __main__():
    args = parse_args()
    init_wandb(args)

    task_name = (
        f"BIOS_model_{args.weights_type}_type_{args.type}_"
        f"{args.testing_balanced}_training_{args.training_data}_"
        f"{args.training_balanced}_seed_{args.seed}_layer_{args.layer}"
    )

    run_MDL_probing(args, load_bias_in_bios_vectors, task_name, shuffle=True)


if __name__ == "__main__":
    __main__()
