# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from unet.model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image


class Predictor(BasePredictor):

    def load(self, checkpoint):
        model = UNET()
        model.load_state_dict(checkpoint["state_dict"])
        return model
    
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        print("==> loading model")
        self.model = self.load(torch.load("checkpoint/checkpoint.pth.tar", map_location=torch.device('cuda'))) 
        self.model.to("cuda")

        self.transform = A.Compose(
            [
                A.Resize(height=512, width=512),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ])

    def predict(
        self,
        image: Path = Input(description="input Image")
    ) -> Path:
        """Run a single prediction on the model"""
        image = Image.open(image).convert("RGBA")
        processed_input = self.transform(image)
        processed_input = processed_input.to("cuda")
        output = self.model(processed_input)
        output = (output >0.5)
        output = output.squeeze(0).squeeze(0)
        output = output.numpy()
        result_image = Image.fromarray(output)
        result_image.resize(image.size)
        output_path = f"/tmp/out-0.png"
        result_image.save(output_path)
        return  Path(output_path)
