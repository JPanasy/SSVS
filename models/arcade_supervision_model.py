import torch
import torch.nn.functional as F
from collections import OrderedDict
from .base_model import BaseModel
from . import networks

class ARCADESupervisionModel(BaseModel):
    """
    Supervised pretraining model for ARCADE data.
    
    This model uses only the generator G_A with supervised losses to learn vessel
    and stenosis segmentation from labeled ARCADE data. After pretraining, G_A
    weights can be transferred to downstream segmentation or classification work.
    
    Loss functions:
    - Focal Tversky for imbalanced vessel-region segmentation
    - clDice for centerline/topology preservation
    - Optional BCE fallback
    - Optional Kendall homoscedastic uncertainty weighting
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(dataset_mode='arcade')
        parser.set_defaults(netG='unet_512')
        parser.set_defaults(no_dropout=False)  # May want dropout for regularization
        
        if is_train:
            parser.add_argument('--lambda_vessel', type=float, default=1.0,
                                help='Weight for vessel mask loss')
            parser.add_argument('--lambda_stenosis', type=float, default=0.5,
                                help='Weight for stenosis mask loss (if learning both)')
            parser.add_argument('--use_focal_loss', action='store_true',
                                help='Use focal loss instead of BCE for imbalanced segmentation')
            parser.add_argument('--seg_loss', type=str, default='bce_focal_tversky_cldice',
                                choices=['bce', 'focal', 'focal_tversky', 'cldice',
                                         'focal_tversky_cldice', 'bce_focal_tversky_cldice',
                                         'scheduled_hybrid_boundary'],
                                help='Segmentation loss for ARCADE supervised pretraining')
            parser.add_argument('--use_boundary_loss', action='store_true',
                                help='Use precomputed distance-map boundary loss')
            parser.add_argument('--tversky_alpha', type=float, default=0.7,
                                help='False-positive penalty for Tversky loss')
            parser.add_argument('--tversky_beta', type=float, default=0.3,
                                help='False-negative penalty for Tversky loss')
            parser.add_argument('--focal_tversky_gamma', type=float, default=0.75,
                                help='Focal exponent for Focal Tversky loss')
            parser.add_argument('--focal_alpha', type=float, default=0.25,
                                help='Alpha for standard focal loss')
            parser.add_argument('--focal_gamma', type=float, default=2.0,
                                help='Gamma for standard focal loss')
            parser.add_argument('--cldice_iter', type=int, default=10,
                                help='Soft skeletonization iterations for clDice')
            parser.add_argument('--lambda_bce', type=float, default=0.5,
                                help='BCE weight in bce_focal_tversky_cldice loss')
            parser.add_argument('--lambda_focal_tversky', type=float, default=0.3,
                                help='Focal Tversky weight in combined segmentation loss')
            parser.add_argument('--lambda_cldice', type=float, default=0.2,
                                help='Fixed clDice weight when uncertainty weighting is disabled')
            parser.add_argument('--bce_warmup_epochs', type=int, default=5,
                                help='Use pure BCE for this many initial epochs before combined loss')
            parser.add_argument('--regional_warmup_epochs', type=int, default=20,
                                help='Scheduled hybrid: use BCE+Focal+Tversky only through this epoch')
            parser.add_argument('--structural_warmup_epochs', type=int, default=50,
                                help='Scheduled hybrid: add clDice after regional warmup through this epoch')
            parser.add_argument('--boundary_ramp_epochs', type=int, default=50,
                                help='Scheduled hybrid: ramp contour/topology weights over this many epochs after structural warmup')
            parser.add_argument('--lambda_bce_final', type=float, default=0.5,
                                help='Scheduled hybrid: final BCE weight after ramp')
            parser.add_argument('--lambda_focal', type=float, default=1.0,
                                help='Scheduled hybrid: focal loss weight')
            parser.add_argument('--lambda_tversky', type=float, default=1.0,
                                help='Scheduled hybrid: Tversky loss weight')
            parser.add_argument('--lambda_cldice_final', type=float, default=0.5,
                                help='Scheduled hybrid: final clDice weight after ramp')
            parser.add_argument('--lambda_boundary_final', type=float, default=0.5,
                                help='Scheduled hybrid: final boundary loss weight after ramp')
            parser.add_argument('--use_uncertainty_weighting', action='store_true',
                                help='Use Kendall homoscedastic uncertainty weighting across loss terms')
        
        return parser

    def __init__(self, opt):
        """Initialize supervised model."""
        BaseModel.__init__(self, opt)
        
        # Loss names for logging
        self.loss_names = ['vessel', 'bce', 'focal', 'tversky', 'focal_tversky', 'cldice', 'boundary']
        if self.isTrain and opt.use_uncertainty_weighting:
            self.loss_names.extend(['log_var_focal_tversky', 'log_var_cldice'])
        if opt.arcade_mask_type in ['stenosis', 'both']:
            self.loss_names.append('stenosis')
        
        # Visual names (what to display)
        self.visual_names = ['image', 'vessel_pred', 'vessel_mask']
        if opt.arcade_mask_type in ['stenosis', 'both']:
            self.visual_names.extend(['stenosis_pred', 'stenosis_mask'])
        
        # Model names (what to save/load)
        if self.isTrain:
            self.model_names = ['G_A']  # Only generator for supervised segmentation
        else:
            self.model_names = ['G_A']
        
        # Define generator
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                        opt.norm, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            # Generators in this codebase end with tanh, so convert their
            # outputs to probabilities before applying segmentation losses.
            self.criterionBCE = torch.nn.BCELoss()
            self.criterionFocal = FocalLoss(alpha=opt.focal_alpha, gamma=opt.focal_gamma)
            self.criterionTversky = TverskyLoss(
                alpha=opt.tversky_alpha,
                beta=opt.tversky_beta,
            )
            self.criterionFocalTversky = FocalTverskyLoss(
                alpha=opt.tversky_alpha,
                beta=opt.tversky_beta,
                gamma=opt.focal_tversky_gamma,
            )
            self.criterionClDice = SoftClDiceLoss(iterations=opt.cldice_iter)
            self.criterionBoundary = BoundaryLoss()
            
            # Optional focal loss for imbalanced data
            self.criterion_seg = self.criterionFocal if opt.use_focal_loss else self.criterionBCE

            if opt.use_uncertainty_weighting:
                self.netUncertainty = HomoscedasticUncertaintyWeights().to(self.device)
                self.model_names.append('Uncertainty')
                uncertainty_params = list(self.netUncertainty.parameters())
            else:
                uncertainty_params = []
            
            # Optimizer (only for generator)
            self.optimizer_G = torch.optim.Adam(
                list(self.netG_A.parameters()) + uncertainty_params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
            )
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input from dataloader."""
        self.image = input['image'].to(self.device)
        
        if 'vessel_mask' in input:
            self.vessel_mask = input['vessel_mask'].to(self.device)
        else:
            self.vessel_mask = None

        if 'vessel_boundary_map' in input:
            self.vessel_boundary_map = input['vessel_boundary_map'].to(self.device)
        else:
            self.vessel_boundary_map = None
        
        if 'stenosis_mask' in input:
            self.stenosis_mask = input['stenosis_mask'].to(self.device)
        else:
            self.stenosis_mask = None
        
        self.image_paths = input.get('paths', [''])

    def forward(self):
        """Run forward pass."""
        raw_output = self.netG_A(self.image)
        self.vessel_pred = self._to_probability(raw_output)
        self.vessel_pred_viz = self.vessel_pred
        self.stenosis_pred = self.vessel_pred
        self.stenosis_pred_viz = self.stenosis_pred

    def backward_G(self):
        """Calculate generator loss (segmentation loss)."""
        self.loss_vessel = 0
        self.loss_stenosis = 0
        self.loss_bce = 0
        self.loss_focal = 0
        self.loss_tversky = 0
        self.loss_focal_tversky = 0
        self.loss_cldice = 0
        self.loss_boundary = 0
        if self.opt.use_uncertainty_weighting:
            self.loss_log_var_focal_tversky = self.netUncertainty.log_var_focal_tversky
            self.loss_log_var_cldice = self.netUncertainty.log_var_cldice
        
        # Vessel segmentation loss
        if self.vessel_mask is not None:
            self.loss_vessel = self._segmentation_loss(
                self.vessel_pred,
                self.vessel_mask,
                self.vessel_boundary_map,
            ) * self.opt.lambda_vessel
        
        # Stenosis segmentation loss (multi-task learning)
        if self.stenosis_mask is not None:
            self.loss_stenosis = self.criterionBCE(self.stenosis_pred, self.stenosis_mask) * self.opt.lambda_stenosis
        
        # Total loss
        self.loss_G = self.loss_vessel + self.loss_stenosis
        
        if self.loss_G > 0:
            self.loss_G.backward()
        
        return self.loss_G

    def _segmentation_loss(self, pred, target, boundary_map=None):
        """Combine selected segmentation losses."""
        self.loss_bce = self.criterionBCE(pred, target)

        if self.opt.seg_loss == 'scheduled_hybrid_boundary':
            return self._scheduled_hybrid_boundary_loss(pred, target, boundary_map)

        # Stabilize early U-Net training with strict pixel-wise supervision.
        if self.opt.bce_warmup_epochs > 0 and self.current_epoch <= self.opt.bce_warmup_epochs:
            return self.loss_bce

        if self.opt.seg_loss == 'bce':
            return self.loss_bce
        if self.opt.seg_loss == 'focal':
            self.loss_focal = self.criterionFocal(pred, target)
            return self.loss_focal

        focal_tversky_losses = [
            'focal_tversky',
            'focal_tversky_cldice',
            'bce_focal_tversky_cldice',
        ]
        cldice_losses = [
            'cldice',
            'focal_tversky_cldice',
            'bce_focal_tversky_cldice',
        ]

        if self.opt.seg_loss in focal_tversky_losses:
            self.loss_focal_tversky = self.criterionFocalTversky(pred, target)
        if self.opt.seg_loss in cldice_losses:
            self.loss_cldice = self.criterionClDice(pred, target)

        if self.opt.seg_loss == 'focal_tversky':
            return self.loss_focal_tversky
        if self.opt.seg_loss == 'cldice':
            return self.loss_cldice

        if self.opt.seg_loss == 'bce_focal_tversky_cldice':
            return (
                self.opt.lambda_bce * self.loss_bce
                + self.opt.lambda_focal_tversky * self.loss_focal_tversky
                + self.opt.lambda_cldice * self.loss_cldice
            )

        if self.opt.use_uncertainty_weighting:
            return (
                self._uncertainty_weighted(self.loss_focal_tversky, self.netUncertainty.log_var_focal_tversky)
                + self._uncertainty_weighted(self.loss_cldice, self.netUncertainty.log_var_cldice)
            )

        return self.loss_focal_tversky + self.opt.lambda_cldice * self.loss_cldice

    def _scheduled_hybrid_boundary_loss(self, pred, target, boundary_map):
        weights = self._scheduled_hybrid_weights()
        if weights['boundary'] > 0 and boundary_map is None:
            raise RuntimeError(
                "Boundary loss is active but no vessel_boundary_map was provided. "
                "Run precompute_arcade_distance_maps.py and pass "
                "--use_boundary_loss --boundary_map_dir <dir>."
            )
        self.loss_focal = self.criterionFocal(pred, target)
        self.loss_tversky = self.criterionTversky(pred, target)

        total = (
            weights['bce'] * self.loss_bce
            + weights['focal'] * self.loss_focal
            + weights['tversky'] * self.loss_tversky
        )

        if weights['cldice'] > 0:
            self.loss_cldice = self.criterionClDice(pred, target)
            total = total + weights['cldice'] * self.loss_cldice

        if weights['boundary'] > 0:
            self.loss_boundary = self.criterionBoundary(pred, boundary_map)
            total = total + weights['boundary'] * self.loss_boundary

        return total

    def _scheduled_hybrid_weights(self):
        epoch = max(1, int(self.current_epoch))
        if epoch <= self.opt.regional_warmup_epochs:
            return {
                'bce': 1.0,
                'focal': self.opt.lambda_focal,
                'tversky': self.opt.lambda_tversky,
                'cldice': 0.0,
                'boundary': 0.0,
            }

        if epoch <= self.opt.structural_warmup_epochs:
            return {
                'bce': 1.0,
                'focal': self.opt.lambda_focal,
                'tversky': self.opt.lambda_tversky,
                'cldice': self.opt.lambda_cldice,
                'boundary': 0.0,
            }

        ramp_epoch = epoch - self.opt.structural_warmup_epochs
        ramp = min(1.0, ramp_epoch / max(1, self.opt.boundary_ramp_epochs))
        return {
            'bce': 1.0 + ramp * (self.opt.lambda_bce_final - 1.0),
            'focal': self.opt.lambda_focal,
            'tversky': self.opt.lambda_tversky,
            'cldice': self.opt.lambda_cldice + ramp * (self.opt.lambda_cldice_final - self.opt.lambda_cldice),
            'boundary': ramp * self.opt.lambda_boundary_final,
        }

    @staticmethod
    def _uncertainty_weighted(loss, log_var):
        """Kendall homoscedastic uncertainty weighting for one task loss."""
        return torch.exp(-log_var) * loss + log_var

    def optimize_parameters(self):
        """Single optimization step (only generator)."""
        self.forward()
        
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):
        """Return visualization images."""
        visual_dict = OrderedDict()
        
        # Add input image
        visual_dict['image'] = self.image
        
        # Add vessel predictions and ground truth
        if self.vessel_mask is not None:
            visual_dict['vessel_pred'] = self.vessel_pred_viz
            visual_dict['vessel_mask'] = self.vessel_mask
        
        # Add stenosis if available
        if self.stenosis_mask is not None and 'stenosis_pred' in self.visual_names:
            visual_dict['stenosis_pred'] = self.stenosis_pred_viz
            visual_dict['stenosis_mask'] = self.stenosis_mask
        
        return visual_dict

    @staticmethod
    def _to_probability(pred):
        """Map tanh generator output from [-1, 1] to stable [0, 1] probabilities."""
        return ((pred + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)

class FocalLoss(torch.nn.Module):
    """Standard binary focal loss on probabilities."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1.0 - pred) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = alpha_t * (1.0 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()


class TverskyLoss(torch.nn.Module):
    """Tversky loss for controlling false positives versus false negatives."""

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        true_positive = (pred * target).sum(dim=1)
        false_positive = (pred * (1.0 - target)).sum(dim=1)
        false_negative = ((1.0 - pred) * target).sum(dim=1)
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return (1.0 - tversky).mean()


class BoundaryLoss(torch.nn.Module):
    """Boundary loss using a precomputed signed distance map."""

    def forward(self, pred, signed_distance_map):
        return (pred * signed_distance_map).mean()


class FocalTverskyLoss(torch.nn.Module):
    """Focal Tversky loss for imbalanced binary segmentation."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        true_positive = (pred * target).sum(dim=1)
        false_positive = (pred * (1.0 - target)).sum(dim=1)
        false_negative = ((1.0 - pred) * target).sum(dim=1)
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return torch.pow(1.0 - tversky, self.gamma).mean()


class HomoscedasticUncertaintyWeights(torch.nn.Module):
    """Learned log variances for Kendall uncertainty loss weighting."""

    def __init__(self):
        super().__init__()
        self.log_var_focal_tversky = torch.nn.Parameter(torch.zeros(1))
        self.log_var_cldice = torch.nn.Parameter(torch.zeros(1))


class SoftClDiceLoss(torch.nn.Module):
    """Differentiable clDice loss based on soft skeletonization."""

    def __init__(self, iterations=10, smooth=1e-6):
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, pred, target):
        pred_skeleton = self._soft_skeleton(pred)
        target_skeleton = self._soft_skeleton(target)
        topology_precision = (pred_skeleton * target).sum(dim=(1, 2, 3)) / (
            pred_skeleton.sum(dim=(1, 2, 3)) + self.smooth
        )
        topology_sensitivity = (target_skeleton * pred).sum(dim=(1, 2, 3)) / (
            target_skeleton.sum(dim=(1, 2, 3)) + self.smooth
        )
        cl_dice = (2.0 * topology_precision * topology_sensitivity) / (
            topology_precision + topology_sensitivity + self.smooth
        )
        return (1.0 - cl_dice).mean()

    def _soft_skeleton(self, image):
        skeleton = F.relu(image - self._soft_open(image))
        image_eroded = image
        for _ in range(self.iterations):
            image_eroded = self._soft_erode(image_eroded)
            opened = self._soft_open(image_eroded)
            delta = F.relu(image_eroded - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton

    @staticmethod
    def _soft_erode(image):
        erode_h = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
        erode_w = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.min(erode_h, erode_w)

    def _soft_dilate(self, image):
        return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)

    def _soft_open(self, image):
        return self._soft_dilate(self._soft_erode(image))

