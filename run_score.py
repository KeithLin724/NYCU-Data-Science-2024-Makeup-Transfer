import os
from pathlib import Path

import click
from tqdm import tqdm
from PIL import Image

from ai import MakeupTransfer, Args, compute_metrics


class Tester:
    TEST_FILE = "./makeup_test.txt", "./nomakeup_test.txt"
    OUTPUT_ROOT = "./out"

    def __init__(self):
        return

    def __get_test_order(self, dataset: str = "mtdataset"):
        with (
            open(Tester.TEST_FILE[0], mode="r") as f_makeup,
            open(Tester.TEST_FILE[1], mode="r") as f_non_makeup,
        ):
            path_add = lambda file: f"./{dataset}/images/{file}"

            makeup_list = [path_add(item.strip()) for item in f_makeup.readlines()]
            non_makeup_list = [
                path_add(item.strip()) for item in f_non_makeup.readlines()
            ]

        return list(zip(non_makeup_list, makeup_list))

    def test(
        self,
        checkpoint: str,
        output_sub_folder: str,
        run_mark: bool = False,
        model_type: str = "branch",
    ):
        model = MakeupTransfer(checkpoint, model_type=model_type)
        test_data_order = self.__get_test_order()

        output_folder = os.path.join(Tester.OUTPUT_ROOT, output_sub_folder)

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        for index, item in enumerate(
            tqdm(test_data_order, desc="Generating Test image...", unit="img")
        ):
            non_makeup_path, makeup_path = item

            non_makeup_image, makeup_image = (
                Image.open(non_makeup_path),
                Image.open(makeup_path),
            )

            x, y = model(non_makeup_image, makeup_image)

            x.save(os.path.join(output_folder, f"pred_{index}.png"))

        if not run_mark:
            return

        print("run mark")

        args_pass = Args(generate_root=output_folder)
        result_mark = compute_metrics(args=args_pass)

        print(result_mark)

        return


@click.command()
@click.option("--checkpoint", type=click.STRING, help="loading checkpoint")
@click.option("--out-folder", default="mt", type=click.STRING, help="test output")
@click.option("--run-mark", default=True, type=click.BOOL, help="run test mark")
def main(checkpoint: str, out_folder: str, run_mark: bool):
    tester = Tester()
    tester.test(checkpoint=checkpoint, output_sub_folder=out_folder, run_mark=run_mark)
    return


if __name__ == "__main__":
    main()
