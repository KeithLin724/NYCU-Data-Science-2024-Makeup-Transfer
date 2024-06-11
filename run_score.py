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
        add_resize_128: bool = False,
        model_type: str = "branch",
    ):
        model = MakeupTransfer(checkpoint, model_type=model_type)
        test_data_order = self.__get_test_order()

        output_folder = os.path.join(Tester.OUTPUT_ROOT, output_sub_folder)

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        if add_resize_128:
            output_folder_128 = os.path.join(
                Tester.OUTPUT_ROOT,
                f"{output_sub_folder}_128",
            )

            Path(output_folder_128).mkdir(parents=True, exist_ok=True)

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

            if not add_resize_128:
                continue

            x_128 = x.resize((128, 128))
            x_128.save(os.path.join(output_folder_128, f"pred_{index}.png"))

        if not run_mark:
            return

        print("run mark")

        args_pass = Args(generate_root=output_folder)
        result_mark = compute_metrics(args=args_pass)

        print(result_mark)

        if not add_resize_128:
            return

        print("run mark 128")

        args_pass = Args(generate_root=output_folder_128)
        result_mark = compute_metrics(args=args_pass)

        print(result_mark)

        return


@click.command()
@click.option("--checkpoint", type=click.STRING, help="loading checkpoint")
@click.option("--out-folder", default="mt", type=click.STRING, help="test output")
@click.option("--run-mark", default=True, type=click.BOOL, help="run test mark")
@click.option("--add_128", default=False, type=click.BOOL, help="add 128 images folder")
def main(checkpoint: str, out_folder: str, run_mark: bool, add_128: bool):
    tester = Tester()
    tester.test(
        checkpoint=checkpoint,
        output_sub_folder=out_folder,
        run_mark=run_mark,
        add_resize_128=add_128,
    )
    return


if __name__ == "__main__":
    main()
