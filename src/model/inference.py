import os

from src.model.modules.imagecraft import ImageCraft


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "imagecraft"
import argparse

from PIL import Image


def run(args):

    model = ImageCraft.from_pretrained("nsandiman/imagecraft-ft-co-224")

    image = Image.open(args.image_path)
    image.load()

    audio_file = model.generate(image, args.output_type)
    print(audio_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Running inference on imagecraft model."
    )
    parser.add_argument("--image_path", type=str, default="media/images/1.jpeg")
    parser.add_argument("--output_type", type=str, default="file")

    args = parser.parse_args()

    run(args)
