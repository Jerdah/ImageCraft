from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CaptionDataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        caption = item["caption"]

        image = item["image"].convert("RGB")
        image = self.transform(image)
        if isinstance(caption, list):
            caption = caption[0]

        return image, caption
