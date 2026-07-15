import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class GLMAENet(nn.Module):
    """Compact global-local masked autoencoder for vessel pretraining."""

    def __init__(self, input_nc=1, embed_dim=256):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, input_nc, 1, 1))
        self.encoder = nn.Sequential(
            ConvBlock(input_nc, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, embed_dim, stride=2),
            ConvBlock(embed_dim, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, input_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, image, mask):
        masked_image = image * (1.0 - mask) + self.mask_token * mask
        return self.decoder(self.encoder(masked_image))


class SwinMAENet(nn.Module):
    """Swin-backed masked autoencoder using torchvision's Swin Transformer."""

    def __init__(self, input_nc=1, output_nc=1):
        super().__init__()
        try:
            from torchvision.models import swin_t
        except ImportError as exc:
            raise ImportError("swin_mae requires torchvision with swin_t support.") from exc

        self.mask_token = nn.Parameter(torch.zeros(1, input_nc, 1, 1))
        self.encoder = swin_t(weights=None).features
        self.encoder[0][0] = nn.Conv2d(input_nc, 96, kernel_size=4, stride=4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.ConvTranspose2d(48, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, image, mask):
        masked_image = image * (1.0 - mask) + self.mask_token * mask
        features = self.encoder(masked_image)
        if features.shape[1] != 768:
            features = features.permute(0, 3, 1, 2).contiguous()
        reconstruction = self.decoder(features)
        if reconstruction.shape[-2:] != image.shape[-2:]:
            reconstruction = F.interpolate(
                reconstruction,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction


class MaePretrainModel(BaseModel):
    """Masked autoencoder pretraining with GL-MAE or Swin-MAE backbones."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(dataset_mode="arcade")
        parser.set_defaults(no_dropout=True)
        parser.add_argument(
            "--mae_arch",
            type=str,
            default="gl_mae",
            choices=["gl_mae", "swin_mae"],
            help="Masked autoencoder backbone",
        )
        parser.add_argument("--mae_mask_ratio", type=float, default=0.6)
        parser.add_argument("--mae_patch_size", type=int, default=16)
        parser.add_argument("--mae_local_weight", type=float, default=0.5)
        parser.add_argument("--sgd_momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["recon", "global", "local"]
        self.visual_names = ["image", "masked_image", "reconstruction", "mask"]
        self.model_names = ["MAE"]

        if opt.mae_arch == "swin_mae":
            self.netMAE = SwinMAENet(opt.input_nc, opt.output_nc).to(self.device)
        else:
            self.netMAE = GLMAENet(opt.input_nc).to(self.device)

        if self.isTrain:
            self.optimizer_MAE = torch.optim.SGD(
                self.netMAE.parameters(),
                lr=opt.lr,
                momentum=opt.sgd_momentum,
                weight_decay=opt.weight_decay,
            )
            self.optimizers.append(self.optimizer_MAE)

    def set_input(self, input_data):
        if "image" in input_data:
            self.image = input_data["image"].to(self.device)
            self.image_paths = input_data.get("paths", [""])
        elif "A" in input_data:
            self.image = input_data["A"].to(self.device)
            self.image_paths = input_data.get("A_paths", [""])
        else:
            raise KeyError("MaePretrainModel expects input key 'image' or 'A'.")

    def forward(self):
        self.mask = self._make_mask(self.image)
        self.masked_image = self.image * (1.0 - self.mask)
        self.reconstruction = self.netMAE(self.image, self.mask)

    def backward_MAE(self):
        masked_pixels = self.mask.sum().clamp_min(1.0)
        self.loss_global = (((self.reconstruction - self.image) ** 2) * self.mask).sum() / masked_pixels

        local_weight = self._make_local_weight(self.image)
        local_mask = self.mask * local_weight
        local_pixels = local_mask.sum().clamp_min(1.0)
        self.loss_local = (((self.reconstruction - self.image) ** 2) * local_mask).sum() / local_pixels

        self.loss_recon = self.loss_global + self.opt.mae_local_weight * self.loss_local
        self.loss_recon.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_MAE.zero_grad()
        self.backward_MAE()
        self.optimizer_MAE.step()

    def _make_mask(self, image):
        batch_size, _, height, width = image.shape
        patch_size = self.opt.mae_patch_size
        mask_h = max(1, height // patch_size)
        mask_w = max(1, width // patch_size)
        mask = torch.rand(batch_size, 1, mask_h, mask_w, device=image.device)
        mask = (mask < self.opt.mae_mask_ratio).float()
        return F.interpolate(mask, size=(height, width), mode="nearest")

    @staticmethod
    def _make_local_weight(image):
        batch_size, _, height, width = image.shape
        weight = torch.zeros(batch_size, 1, height, width, device=image.device)
        crop_h = max(1, height // 2)
        crop_w = max(1, width // 2)
        for batch_index in range(batch_size):
            top = torch.randint(0, height - crop_h + 1, (1,), device=image.device).item()
            left = torch.randint(0, width - crop_w + 1, (1,), device=image.device).item()
            weight[batch_index, :, top : top + crop_h, left : left + crop_w] = 1.0
        return weight
