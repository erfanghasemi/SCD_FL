from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
import torch

import utils as utils

import warnings
warnings.filterwarnings("ignore")

SERVER_NUM_ROUNDS = 5
SERVER_LAST_UNFREEZE_LAYERS_COUNT = 3
SERVER_BATCH_SIZE_EVAL = 32
SERVER_FRACTION_FIT = 1.0
SERVER_FRACTION_EVALUATE = 1.0
SERVER_MIN_FIT_CLIENTS = 1
SERVER_MIN_EVALUATE_CLIENTS = 1
SERVER_MIN_AVAILABLE_CLIENTS = 1
SERVER_TOY_SAMPLES_COUNT = 4

CLIENT_BATCH_SIZE_LEARNING = 32
CLIENT_BATCH_SIZE_TEST = 32

CLIENT_LOCAL_EPOCHS_STEP_ZERO = 1
CLIENT_LOCAL_EPOCHS_STEP_ONE = 2
CLIENT_EVAL_STEPS_ZERO = 2
CLIENT_EVAL_STEPS_ONE = 3

SERVER_MODEL_CHECKPOINTS_DIR_PATH = "./server_checkpoints"


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": CLIENT_BATCH_SIZE_LEARNING,
        "local_epochs": CLIENT_LOCAL_EPOCHS_STEP_ZERO if server_round < 2 else CLIENT_LOCAL_EPOCHS_STEP_ONE,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    eval_steps = CLIENT_EVAL_STEPS_ZERO if server_round < 4 else CLIENT_EVAL_STEPS_ONE
    batch_size = CLIENT_BATCH_SIZE_TEST
    return {
        "eval_steps": eval_steps,
        "batch_size": batch_size,
    }

def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    _, testset, _ = utils.load_data()

    n_test = len(testset)
    if toy:
        # use only TOY_SAMPLES_COUNT samples as test set
        testset = torch.utils.data.Subset(testset, range(n_test - SERVER_TOY_SAMPLES_COUNT, n_test))
    else:
        # Use the all test examples as a test set
        testset = torch.utils.data.Subset(testset, range(n_test))

    testLoader = DataLoader(testset, batch_size=SERVER_BATCH_SIZE_EVAL, shuffle=True)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        # utils.save_checkpoint(model, SERVER_MODEL_CHECKPOINTS_DIR_PATH)
        loss, accuracy = utils.test(model, testLoader)
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy:.6f}")
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `--toy`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only TOY_SAMPLES_COUNT datasamples for evaluation. \
            Useful for testing purposes. Default: False",
    )

    args = parser.parse_args()

    model = utils.load_model(layer_count=SERVER_LAST_UNFREEZE_LAYERS_COUNT, device="cpu")

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=SERVER_FRACTION_FIT,
        fraction_evaluate=SERVER_FRACTION_EVALUATE,
        min_fit_clients=SERVER_MIN_FIT_CLIENTS,
        min_evaluate_clients=SERVER_MIN_EVALUATE_CLIENTS,
        min_available_clients=SERVER_MIN_AVAILABLE_CLIENTS,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=SERVER_NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
