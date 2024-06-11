import argparse
import glob
import math
import os

import tabulate
import torch
import torch.nn
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

import torch_fidelity

ROOT = {
    "mt": "mtdataset/images",
    "mt_removal": "mt_removal",
}


class TransformPILtoRGBTensor:
    def __call__(self, img):
        return F.pil_to_tensor(img)


class ImagesPathDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        self.transforms = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.CenterCrop((128, 128)),
                TransformPILtoRGBTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img


class CheckImagesPathDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        ori_file=None,
        target_files=None,
        order_files=None,
    ):
        self.files = []
        self.mask_list = None
        if ori_file is None:
            for file in order_files:
                num = int(os.path.basename(file).split(".")[0].split("_")[-1])
                self.files.append(os.path.join(root, f"pred_{num}.png"))
        else:
            with open(ori_file, "r") as f:
                line_of_file = f.readlines()
            for file in target_files:
                num = int(os.path.basename(file).split(".")[0].split("_")[-1])
                self.files.append(os.path.join(root, line_of_file[num].strip()))
        self.transforms = self.transforms = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.CenterCrop((128, 128)),
                TransformPILtoRGBTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img


def main(args):
    table = []
    for target in [args.generate_root]:
        if not os.path.exists(target):
            continue
        target_file_sorted = sorted(
            list(glob.glob(f"{target}/*.png")),
            key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]),
        )
        target_non_makeup_img = []
        with open(args.non_makeup_file, "r") as f:
            for line in f:
                target_non_makeup_img.append(os.path.join(ROOT["mt"], line.strip()))
        precision = torch_fidelity.calculate_metrics(
            input1=ImagesPathDataset(files=target_file_sorted),
            input2=ImagesPathDataset(files=target_non_makeup_img),
            input3=CheckImagesPathDataset(
                root=ROOT[args.type],
                ori_file=args.makeup_file,
                target_files=target_file_sorted,
            ),
            input5=CheckImagesPathDataset(
                root=ROOT[f"{args.type}_removal"],
                order_files=target_file_sorted,
            ),
            fid=False,
            kid=False,
            prc=True,
            device="cuda",
            verbose=False,
        )["precision"]

        recall = torch_fidelity.calculate_metrics(
            input1=ImagesPathDataset(
                files=target_file_sorted,
            ),
            input2=CheckImagesPathDataset(
                root=ROOT[args.type],
                ori_file=args.makeup_file,
                target_files=target_file_sorted,
            ),
            input3=CheckImagesPathDataset(  # Use mt to remove original non-makeup feature
                root=ROOT["mt"],
                ori_file=args.non_makeup_file,
                target_files=target_file_sorted,
            ),
            input4=CheckImagesPathDataset(
                root=ROOT[
                    f"{args.type}_removal"
                ],  # Use makeup to non-makeup image to remove non-makeup feature
                order_files=target_file_sorted,
            ),
            fid=False,
            kid=False,
            prc=True,
            device="cuda",
            verbose=False,
        )["recall"]
        table.append(
            [
                os.path.basename(target),
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{math.sqrt(precision * recall):.3f}",
            ]
        )
    print(
        tabulate.tabulate(
            table,
            headers=["Approach", "Precision", "Recall", "Overall"],
            tablefmt="grid",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute precision and recall")
    parser.add_argument(
        "--generate-root",
        help="File for the generated image",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--non-makeup-file",
        help="File denoting the order of non-makeup image",
        default="nomakeup_test.txt",
    )
    parser.add_argument(
        "--makeup-file",
        help="File denoting the order of makeup image",
        default="makeup_test.txt",
    )
    parser.add_argument(
        "--type",
        help="Type of the approach",
        choices=["mt"],
        default="mt",
    )
    args = parser.parse_args()
    main(args)
