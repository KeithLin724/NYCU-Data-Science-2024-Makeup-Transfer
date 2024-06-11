from PIL.Image import Image

import cv2
import numpy as np

import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from module import Generator_makeup, Generator_branch
from .config_class import TrainingConfig
from .math_tools import de_norm, to_var


class MakeupTransfer:
    def __init__(self, model_path: str, model_type: str = "branch") -> None:

        self.model = MakeupTransfer.load_model(path=model_path, model_type=model_type)

        self.model = self.model.cuda()

        self.transform = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ToTensorV2(),
            ],
            additional_targets={"image_B": "image"},
        )

        self.__tensor_to_pil = transforms.ToPILImage()

        self.pil_to_tensor = lambda *images: [
            cv2.resize(np.array(image), (256, 256)) for image in images
        ]

        self.tensor_to_pil = lambda *images: [
            self.__tensor_to_pil(de_norm(image)) for image in images
        ]

        self.cuda_to_cpu = lambda *tensors: [
            tensor.cpu().squeeze(0) for tensor in tensors
        ]

    @staticmethod
    def load_model(
        path: str,
        config: TrainingConfig = TrainingConfig(),
        model_type: str = "branch",
    ) -> Generator_makeup | Generator_branch:

        model_choose: dict[str, Generator_makeup | Generator_branch] = {
            "makeup": Generator_makeup,
            "branch": Generator_branch,
        }

        model_builder: Generator_makeup | Generator_branch = model_choose[model_type]

        model = model_builder(
            config.g_conv_dim,
            config.g_repeat_num,
        )

        model.load_state_dict(torch.load(path))
        return model

    def to_model_input(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        res = self.transform(image=x, image_B=y)
        x, y = res["image"], res["image_B"]

        x, y = to_var(x, requires_grad=False), to_var(y, requires_grad=False)
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        return x, y

    def __call__(self, x: Image, y: Image) -> tuple[Image, Image]:

        if isinstance(x, Image) and isinstance(y, Image):

            x, y = self.pil_to_tensor(x, y)

            x, y = self.to_model_input(x, y)

        # pass to model
        fake_a, fake_b = self.model(x, y)

        fake_a, fake_b = self.cuda_to_cpu(fake_a, fake_b)

        fake_a, fake_b = self.tensor_to_pil(fake_a, fake_b)

        return fake_a, fake_b
