import utils
import torch

CHECKPOINTS_PATH = "checkpoints"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main() -> None:
    model = utils.load_checkpoint(CHECKPOINTS_PATH, str(DEVICE)) 
       
    image_path = r"lesions_dataset\FL_Test_Dataset\nevus\ISIC_0009925.jpg"
    utils.inference(model, image_path)


if __name__ == "__main__":
    main()
