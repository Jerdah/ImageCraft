import os

from src.model.imagecraft import ImageCraft


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "imagecraft"
import argparse

from PIL import Image


def run():

    model = ImageCraft.from_pretrained("nsandiman/imagecraft-ft-co-224")

    image_path = "media/images/4.jpeg"

    image = Image.open(image_path)
    image.load()

    audio_file = model.generate(image_path, output_type="file")
    print(audio_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Running inference on imagecraft model."
    )
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--model_path", type=str, default="models/imagecraft-ft-fk-224")

    args = parser.parse_args()

    run()
