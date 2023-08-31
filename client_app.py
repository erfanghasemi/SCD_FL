import utils
import torch

CHECKPOINTS_PATH = "checkpoints"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main() -> None:
    model = utils.load_checkpoint(CHECKPOINTS_PATH, str(DEVICE)) 
       
    image_path = r"lesions_dataset\FL_Test_Dataset\melanoma\ISIC_0071541.jpg"
    utils.inference(model, image_path)


if __name__ == "__main__":
    main()
