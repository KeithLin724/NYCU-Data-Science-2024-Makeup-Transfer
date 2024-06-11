from pydantic import BaseModel
from torch import Tensor
import operator as op
from functools import reduce

super_sum = lambda lst: reduce(op.add, lst)


class MakeupMaskLabel(BaseModel):
    eyes_left: int = 1
    eyes_right: int = 6

    nose: int = 8
    upper_lips: int = 9
    lower_lips: int = 13
    hair: int = 12
    neck: int = 10

    eyebrows_left: int = 2
    eyebrows_right: int = 7
    face: int = 4

    ear_left: int = 5
    ear_right: int = 3

    def get_lip(self, mask: Tensor) -> Tensor:
        return super_sum(
            [
                (mask == self.upper_lips).float(),
                (mask == self.lower_lips).float(),
            ]
        )

    def get_skin(self, mask: Tensor) -> Tensor:
        return super_sum(
            [
                (mask == self.face).float(),
                (mask == self.nose).float(),
                (mask == self.neck).float(),
                # (mask == self.ear_left).float(),
                # (mask == self.ear_right).float(),
            ]
        )

    def get_eyes(self, mask: Tensor) -> tuple[Tensor]:
        return (
            super_sum(
                [
                    (mask == self.eyes_left).float(),
                ]
            ),
            super_sum(
                [
                    (mask == self.eyes_right).float(),
                ]
            ),
        )

    def get_face(self, mask: Tensor) -> Tensor:
        return super_sum(
            [
                (mask == self.face).float(),
                (mask == self.nose).float(),
                # (mask == self.eyebrows_left).float(),
                # (mask == self.eyebrows_right).float(),
            ]
        )

    def lips(self, *masks: Tensor):
        return [self.get_lip(mask) for mask in masks]

    def skin(self, *masks: Tensor):
        return [self.get_skin(mask) for mask in masks]

    def eyes(self, *masks: Tensor):
        return [self.get_eyes(mask) for mask in masks]

    def faces(self, *masks: Tensor):
        return [self.get_face(mask) for mask in masks]
