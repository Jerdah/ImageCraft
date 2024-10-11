import os

from src.model.modules.imagecraft import ImageCraft


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "imagecraft"
import argparse


def run(args):

    model = ImageCraft.from_pretrained(args.model_path)

    transcript, speech = model.generate(
        args.image_path, max_tokens=100, do_sample=False, output_type=args.output_type
    )
    print(f"Transcript: {transcript}")
    print(f"Speech file: {speech}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Running inference on imagecraft model."
    )
    parser.add_argument("--image_path", type=str, default="media/images/1.jpeg")
    parser.add_argument("--output_type", type=str, default="file")
    parser.add_argument(
        "--model_path", type=str, default="nsandiman/imagecraft-ft-co-224"
    )

    args = parser.parse_args()

    run(args)
