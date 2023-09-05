import utils as utils
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


CLIENT_VALIDATION_SPLIT = 0.1
CLIENT_TRAIN_TOY_SAMPLES_COUNT = 64
CLIENT_VALIDATION_TOY_SAMPLES_COUNT = 4
CLIENT_TEST_TOY_SAMPLES_COUNT = 4

CLEINT_MODEL_CHECKPOINTS_DIR_PATH = "client_checkpoints"

class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        device: str,
        partition: str,
        toy: bool,
        validation_split: int = CLIENT_VALIDATION_SPLIT,
    ):  
        self.device = device
        self.partition = partition
        self.toy=toy
        self.validation_split = validation_split

    def set_parameters(self, parameters):
        """Loads a modified GoogleNet model and replaces it parameters with the ones
        given."""
        model = utils.load_model(device=self.device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)
        

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        
        # Load trainset and valset
        trainset, _ = utils.load_partition(self.partition)

        n_trainset = len(trainset)
        n_valset = int(len(trainset) * self.validation_split)

        if self.toy:
            valset_indices = torch.randperm(n_trainset)[:CLIENT_VALIDATION_TOY_SAMPLES_COUNT]
            # trainset_indices = torch.randperm(n_trainset)[:CLIENT_TRAIN_TOY_SAMPLES_COUNT]
            
            valset_sampler = SubsetRandomSampler(valset_indices)
            # trainset_sampler = SubsetRandomSampler(trainset_indices)

            # trainLoader = DataLoader(trainset, batch_size=batch_size, sampler=trainset_sampler)
            trainset = torch.utils.data.Subset(trainset, range(0, n_trainset))
            trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valLoader = DataLoader(trainset, batch_size=batch_size, sampler=valset_sampler)
        else:
            valset = torch.utils.data.Subset(trainset, range(0, n_valset))
            trainset = torch.utils.data.Subset(trainset, range(n_valset, n_trainset))
            trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valLoader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        results = utils.train(model, trainLoader, valLoader, epochs, self.device)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)
        utils.save_checkpoint(model, CLEINT_MODEL_CHECKPOINTS_DIR_PATH)
        # Load test dataset for evaluation
        _ , testset = utils.load_partition(self.partition)
        n_testset = len(testset)
        
        if self.toy:
            testset = torch.utils.data.Subset(testset, range(CLIENT_TEST_TOY_SAMPLES_COUNT))
        else:
            testset = torch.utils.data.Subset(testset, range(n_testset))

        # Get config values
        steps: int = config["eval_steps"]
        batch_size: int = config["batch_size"]

        # Evaluate global model parameters on the local test data and return results
        testLoader = DataLoader(testset, batch_size=batch_size)

        loss, accuracy = utils.test(model, testLoader, steps, self.device)
        return float(loss), len(testset), {"accuracy": float(accuracy)}


def client_dry_run(device: str = "cpu"):
    """Weak tests to check whether all client methods are working as
    expected."""

    model = utils.load_model(layer_count=CLIENT_LAST_UNFREEZE_LAYERS_COUNT, device=device)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    client = CifarClient(trainset, testset, device)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 32, "local_epochs": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 5})

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
        help="Specifies the artificial data partition of ISIC 2019 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only CLIENT_TOY_SAMPLES_COUNT datasamples. \
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
    # utils.remove_checkpoints(CLEINT_MODEL_CHECKPOINTS_DIR_PATH)  # comment this line if you need clinet checkpoints in the future