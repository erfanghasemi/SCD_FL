import utils as utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")
LAYERS_COUNT = 3
CHECKPOINTS_PATH = "checkpoints"

class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        device: str,
        partition: str,
        toy: bool,
        validation_split: int = 0.1,
    ):  
        self.device = device
        self.partition = partition
        self.toy=toy
        self.validation_split = validation_split
    def set_parameters(self, parameters):
        """Loads a GoogleNet model and replaces it parameters with the ones
        given."""
        model = utils.load_model(layer_count=LAYERS_COUNT)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        utils.save_checkpoint(model, CHECKPOINTS_PATH)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        self.trainset, self.testset = utils.load_partition(self.partition)
        if self.toy:
            self.trainset = torch.utils.data.Subset(self.trainset, range(50))
            self.testset = torch.utils.data.Subset(self.testset, range(50))

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        results = utils.train(model, trainLoader, valLoader, epochs, self.device)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)
        self.trainset, self.testset = utils.load_partition(self.partition)
        if self.toy:
            self.trainset = torch.utils.data.Subset(self.trainset, range(40))
            self.testset = torch.utils.data.Subset(self.testset, range(40))
        
        # Get config values
        steps: int = config["val_steps"]
        n_valset = int(len(self.trainset) * self.validation_split)
        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))

        # Evaluate global model parameters on the local test data and return results
        valLoader = DataLoader(valset, batch_size=16)

        loss, accuracy = utils.test(model, valLoader, steps, self.device)
        return float(loss), len(valset), {"accuracy": float(accuracy)}


def client_dry_run(device: str = "cpu"):
    """Weak tests to check whether all client methods are working as
    expected."""

    model = utils.load_model(layer_count=LAYERS_COUNT)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    client = CifarClient(trainset, testset, device)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})

    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.dry:
        client_dry_run(device)
    else:


        # Start Flower client
        client = CifarClient(device=device, partition=args.partition, toy=args.toy)

        fl.client.start_numpy_client(server_address="130.185.74.117:8080", client=client)
        


if __name__ == "__main__":
    main()
    utils.remove_checkpoints(CHECKPOINTS_PATH)  # comment this line if you need checkpoints in the future