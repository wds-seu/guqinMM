import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from datamodules import Vocabulary, CROHMEDataModule, ZINCDataModule, JZPDataModule
from models import TSDNetModule
import torchvision.transforms as transforms
from PIL import Image

def load_model(image_path):
    transform = transforms.ToTensor()
    image = Image.open(image_path)
    return transform(image).unsqueeze(0) 

def visualize(image_path):
    
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42)
    node_vocab = Vocabulary("data/wushen_jzp/node_dict.txt")
    edge_vocab = Vocabulary("data/wushen_jzp/edge_dict.txt",
                            use_sos=False,
                            use_eos=False)
    dm = JZPDataModule(node_vocab, edge_vocab)
    dm.setup()
    config_path = "configs/jzp.yml"
    config_f = open(config_path, 'r')
    config_dict = yaml.load(config_f, Loader=yaml.FullLoader)
    config_f.close()
    model = TSDNetModule(**config_dict)

    # Load the image and preprocess it
    image_tensor = load_model(image_path)
    
    # Move the image tensor to the appropriate device
    image_tensor = image_tensor.to(model.device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        print("Predicting on JZP:")
        model.beam_test(dm.test_dataloader(), node_vocab, edge_vocab)

    
    # Process the output as needed (e.g., convert to probabilities, decode, etc.)
    # This step will depend on the specific model and task
    print("Model output:", output)

visualize("./data/jzpdata/images_gray/wushen_1_14.png")
