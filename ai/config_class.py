from pydantic import BaseModel


class Config(BaseModel):
    # setting path
    snapshot_path: str = "./snapshot/"
    pretrained_path: str = "./model/"
    vis_path: str = "./visualization/"
    log_path: str = "./log/"
    data_path: str = "./data/"
    git_ignore: bool = True
    save_num: int = 3
    loss_table: bool = False


# print()
class TrainingConfig(BaseModel):
    task_name: str = ""
    G_LR: float = 2e-5
    D_LR: float = 6e-5
    beta1: float = 0.5
    beta2: float = 0.999
    c_dim: int = 2
    num_epochs: int = 100
    num_epochs_decay: int = 100
    ndis: int = 1
    snapshot_step: int = 260
    log_step: int = 10
    vis_step: int = 260
    batch_size: int = 1
    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_idt: float = 0.5
    img_size: int = 256
    g_conv_dim: int = 64
    d_conv_dim: int = 64
    g_repeat_num: int = 6
    d_repeat_num: int = 3

    # normalization of discriminator, SN means spectrum normalization, none means no normalization
    norm: str = "SN"

    checkpoint: str = ""

    test_model: str = "51_2000"

    # face
    lips: bool = True
    skin: bool = True
    eye: bool = True

    lambda_skin_1: float = 0.1
    lambda_skin_2: float = 0.1

    lambda_his: float = 1
    lambda_eye: float = 1
    lambda_his_lip: float = 1

    lambda_vgg: float = 5e-3

    # 'the classes of makeup to train'
    cls_list: list[str] = ["N", "M"]
    content_layer: str = "r41"
    direct: bool = True

    @property
    def lambda_his_skin_1(self):
        #  args.lambda_his * args.lambda_skin
        return self.lambda_skin_1 * self.lambda_his

    @property
    def lambda_his_skin_2(self):
        #  args.lambda_his * args.lambda_skin
        return self.lambda_skin_2 * self.lambda_his

    @property
    def lambda_his_eye(self):
        # args.lambda_his * args.lambda_eye
        return self.lambda_eye * self.lambda_his


class DatasetConfig(BaseModel):
    # dataset: str
    task_name: str = "MAKEUP"
    img_size: int = 256
    # mask_label: MaskLabelMT = MaskLabelMT()
    using_test_order: bool = False
