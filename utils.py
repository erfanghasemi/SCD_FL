import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import platform
import json
import os
from PIL import Image
import numpy as np

import warnings
warnings.filterwarnings("ignore")

IMAGE_SIZE = 1024  # The size (in pixels) of the images used in the model.
CENTER_CROP_SIZE = 600  # The size (in pixels) for center cropping the image

LOCAL_LEARNING_RATE = 0.05
MOMENTUM = 0.8
WEIGHT_DECAY = 1e-4

MODEL_PATH_SERVER = "./server_checkpoints"
MODEL_PATH_CLIENT = "client_checkpoints"

JSON_PATH = "label_mapping\modified_googlenet_lesions_mapping_labels.json"



"""
This function preprocesses an input image for deep learning model inference.
It follows a series of transformations including resizing and center cropping.
The image is then converted to a tensor and normalized according to specific mean and standard deviation values.
Finally, the preprocessed image is returned as a tensor with a batch dimension.
"""
def preprocess_input(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CENTER_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image


"""
This function extracts and returns the predicted label from the model's output tensor.
It utilizes the provided class index mapping to map the predicted index to a human-readable label.
The resulting label represents the model's prediction for a given input.
"""
def get_predicted_label(output, class_idx):
    results = {}
    output = output.tolist()[0]
    for row in output:
        predicted_index = output.index(row)
        predicted_label = class_idx[str(predicted_index)][1]
        predicted_probability = output[predicted_index]
        results[predicted_label] = predicted_probability

    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    return results



"""
This function is designed for conducting inference with a deep learning model on an input image.
It begins by loading a mapping of class indices from a JSON file, which is crucial for translating
model output. The input image undergoes preprocessing, and the model predicts a label based on it.
The function returns the predicted label, which can be utilized for analysis or user display.
"""
def inference(model, image_path, device):
    # Load the class index mapping from the JSON file
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    json_abs_path = os.path.join(current_dir_path, JSON_PATH)
    with open(json_abs_path) as f:
        class_idx = json.load(f)

    input_image = preprocess_input(image_path)

    model.to(device)
    input_image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        
    # # # Get and print predicted label
    predicted_labels = get_predicted_label(output.cpu().numpy(), class_idx)
    print(f"Predicted Label: {predicted_labels}")

    return predicted_labels
    


""" 
    Generate a new checkpoint filename in each round of federated learning by
    incrementing the version number and save the provided model to the specified
    checkpoint path.
"""
def save_checkpoint(model, checkpoints_dir_path):
    # find the last saved version
    try:
      last_version = int(os.listdir(checkpoints_dir_path)[-1].split('_')[-1].split('.')[0])
    except:
      last_version = 0

    # create save checkpoint path
    model_checkpoint_filename = "MODEL" + "_" + "CHECKPOINT" + "_" + "VERSION" + "_" + str(last_version+1) + ".pth"
    checkpoints_dir_path = os.path.join(checkpoints_dir_path, model_checkpoint_filename)
    
    # save the model 
    torch.save(model, checkpoints_dir_path)
    print("\n checkpoint is saved in path: {}\n".format(checkpoints_dir_path))




""" 
Attempt to load the most recent checkpoint file from the given path.
If successful, load the model onto the specified device (CPU or GPU).
If unsuccessful, print an error message and return -1.
"""
def load_checkpoint(checkpoints_dir_path: str, device):
    try:
      last_checkpoint = os.listdir(checkpoints_dir_path)[-1]
    except:
      print("There is no saved model in this path: {}".format(checkpoints_dir_path))
      return -1

    checkpoints_dir_path = os.path.join(checkpoints_dir_path, last_checkpoint)

    if str(device) == "cuda:0":
        model = torch.load(checkpoints_dir_path)
        print("checkpoint version {} is loaded on GPU".format(last_checkpoint))

    elif str(device) == "cpu" :
        model = torch.load(checkpoints_dir_path, map_location=torch.device('cpu'))
        print("checkpoint version {} is loaded on CPU".format(last_checkpoint))

    else:
        print("Device Type is unknown")
        return -1

    model.to(device)
    return model



"""
This function removes extra checkpoint files in a specified directory.
It first generates a list of checkpoint file paths in the directory.
Then, it iterates through the list and deletes all checkpoint files except the first one.
Finally, it prints a message indicating that extra checkpoints were removed.
"""
def remove_checkpoints(checkpoints_dir_path: str):
    checkpoints = [os.path.join(checkpoints_dir_path ,filename) for filename in os.listdir(checkpoints_dir_path)]
                   
    for checkpoint_file in checkpoints[1:]:
        os.remove(checkpoint_file) 
    print("Extra checkpoints were removed from this path {}.".format(checkpoints_dir_path))




"""
This function loads the ISIC 2019 dataset, which consists of a training and test set.
It applies a series of transformations to preprocess the images, including resizing and normalization.
The function detects the system platform (Linux or Windows) to set the dataset paths accordingly.
If the platform is unsupported, it prints an error message.
The function returns the training set, test set, and the number of examples in each set.
"""
def load_data():
    """Load ISIC 2019 (training and test set)."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(CENTER_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get the system's platform information
    system_platform = platform.system()

    # Check if it's Linux
    if system_platform == "Linux":
        # Further check if it's Linux
        trainset = ImageFolder(root='./lesions_dataset/FL_Training_Dataset', transform=transform)
        testset = ImageFolder(root='./lesions_dataset/FL_Test_Dataset', transform=transform)
        # print("Addresses are correct for Linux sysfile")
    elif system_platform == "Windows":
        trainset = ImageFolder(root='lesions_dataset\FL_Training_Dataset', transform=transform)
        testset = ImageFolder(root='lesions_dataset\FL_Test_Dataset', transform=transform)
        # print("Addresses are correct for Windows sysfile")
    else:
        print("Unsupported operating system detected.")

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples




"""
This function simulates loading a 1/10th partition of the training and test data.
It takes an index `idx` (0 to 9) to specify the partition.
The data is loaded using the `load_data` function, and the number of examples in each set is calculated.
The training and test partitions are then created as subsets of the full datasets.
The function returns the specified training and test partitions for a given index.
"""
def load_partition(idx: int):
    """Load 1/5th of the training and test data to simulate a partition."""
    assert idx in range(5)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / 5)
    n_test = int(num_examples["testset"] / 5)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)





"""
This function trains a neural network on the training set.
It takes the network (`net`), training and validation data loaders (`trainloader` and `valloader`),
the number of training epochs (`epochs`), and a device specification (`device`).
The network is moved to the specified device (CPU or GPU) if available.
Cross-Entropy loss is used as the criterion, and Stochastic Gradient Descent (SGD) is used as the optimizer.
The network is set to training mode, and training proceeds for the specified number of epochs.
After training, the network is moved back to the CPU for testing.
Training and validation performance metrics (loss and accuracy) are computed and returned as a dictionary.
"""
def train(net, trainloader, valloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=LOCAL_LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results



"""
This function validates a neural network on a test or validation set.
It takes the network (`net`), a test data loader (`testloader`), an optional step limit (`steps`), and a device specification (`device`).
The network is moved to the specified device (CPU or GPU) if available.
Cross-Entropy loss is used as the criterion to measure performance.
The function calculates both loss and accuracy of the network.
The network is set to evaluation mode, and the validation loop iterates through the data loader.
The function returns the total loss and accuracy, optionally limited by the specified step count.
"""
def test(net, testloader, steps: int = None, device: str = "cpu"):
    """Validate the network on the test or val set."""
    print("Starting evalutation...")

    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break

    accuracy = correct / len(testloader.dataset)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy




"""
This function unfreezes the final layer(s) of the classifier in a neural network model.
It takes the model (`model`) and an optional parameter (`layer_count`) to specify the number of layers to unfreeze.
By default, it unfreezes the last 3 layers.
The function first freezes all the model's parameters.
It then identifies the last `layer_count` layers and sets their `requires_grad` attribute to True,
allowing them to be trained during subsequent optimization steps.
"""
def unfreeze_classifying_layer(model, layer_count: int = 3):
    """Unfreeze the final layer of the classifier."""
    for param in model.parameters():
        param.requires_grad = False

    all_layers = list(model.children())
    num_layers = len(all_layers)
    last_three_layers = torch.nn.Sequential(*all_layers[num_layers - layer_count:])

    for param in last_three_layers.parameters():
        param.requires_grad = True

    return model




"""
This function loads a pre-trained neural network model based on the operating system and device specified.
It first detects the system's platform (Linux or Windows) to determine the appropriate model path.
Then, it calls the `load_checkpoint` function to load the model from the respective path on the specified device.
If the platform is unsupported, it prints an error message.
Additionally, it can unfreeze the specified number of layers in the classifier using `unfreeze_classifying_layer` (for transfer learning).
The loaded and optionally modified model is returned.
"""
def load_model(device: str, layer_count: int = 3):
    
    # Get the system's platform information
    system_platform = platform.system()

    # Check if it's Ubuntu
    if system_platform == "Linux":
        model = load_checkpoint(MODEL_PATH_SERVER, device)
        # print("Model are correctly loaded for {} on {}".format(system_platform, device))

    elif system_platform == "Windows":
        model = load_checkpoint(MODEL_PATH_CLIENT, device)
        # print("Model are correctly loaded for {} on {}".format(system_platform, device))
    else:
        print("Unsupported operating system detected.")

    # DO NOT USE this line if you need to train whole network
    model = unfreeze_classifying_layer(model, layer_count)
    return model


"""
This function retrieves the parameters of a neural network model.
It takes the model (`model`) as input and returns a list containing the parameters.
The model's state dictionary is iterated, and each parameter is converted to a NumPy array
and moved to the CPU before being added to the list.
The resulting list contains the model's parameters.
"""
def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]