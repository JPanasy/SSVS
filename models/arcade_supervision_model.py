import torch
import itertools
from collections import OrderedDict
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class ARCADESupervisionModel(BaseModel):
    """
    Supervised pretraining model for ARCADE data.
    
    This model uses only the generator G_A with supervised loss (BCE) to learn
    vessel and stenosis segmentation from labeled ARCADE data. After pretraining,
    G_A weights will be transferred to the USSEGModel for cycle-consistency finetuning.
    
    Loss functions:
    - BCE (Binary Cross Entropy) for vessel mask prediction
    - Optional BCE for stenosis if learning both tasks
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(dataset_mode='arcade')
        parser.set_defaults(no_dropout=False)  # May want dropout for regularization
        
        if is_train:
            parser.add_argument('--lambda_vessel', type=float, default=1.0,
                                help='Weight for vessel mask loss')
            parser.add_argument('--lambda_stenosis', type=float, default=0.5,
                                help='Weight for stenosis mask loss (if learning both)')
            parser.add_argument('--use_focal_loss', action='store_true',
                                help='Use focal loss instead of BCE for imbalanced segmentation')
        
        return parser

    def __init__(self, opt):
        """Initialize supervised model."""
        BaseModel.__init__(self, opt)
        
        # Loss names for logging
        self.loss_names = ['vessel']
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
            # Loss functions
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            
            # Optional focal loss for imbalanced data
            if opt.use_focal_loss:
                self.criterionFocal = self._get_focal_loss()
                self.criterion_seg = self.criterionFocal
            else:
                self.criterion_seg = self.criterionBCE
            
            # Optimizer (only for generator)
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                               lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input from dataloader."""
        self.image = input['image'].to(self.device)
        
        if 'vessel_mask' in input:
            self.vessel_mask = input['vessel_mask'].to(self.device)
        else:
            self.vessel_mask = None
        
        if 'stenosis_mask' in input:
            self.stenosis_mask = input['stenosis_mask'].to(self.device)
        else:
            self.stenosis_mask = None
        
        self.image_paths = input.get('paths', [''])

    def forward(self):
        """Run forward pass."""
        # Output ranges from tanh: (-1, 1), convert to (0, 1) for BCE
        raw_output = self.netG_A(self.image)
        self.vessel_pred = raw_output  # Keep raw for BCE loss
        
        # For visualization, convert to (0, 1)
        self.vessel_pred_viz = torch.sigmoid(self.vessel_pred)

    def backward_G(self):
        """Calculate generator loss (segmentation loss)."""
        self.loss_vessel = 0
        self.loss_stenosis = 0
        
        # Vessel segmentation loss
        if self.vessel_mask is not None:
            self.loss_vessel = self.criterion_seg(self.vessel_pred, self.vessel_mask) * self.opt.lambda_vessel
        
        # Stenosis segmentation loss (multi-task learning)
        if self.stenosis_mask is not None:
            stenosis_pred = self.netG_A(self.image)  # Could add branch for multi-task
            self.loss_stenosis = self.criterionBCE(stenosis_pred, self.stenosis_mask) * self.opt.lambda_stenosis
        
        # Total loss
        self.loss_G = self.loss_vessel + self.loss_stenosis
        
        if self.loss_G > 0:
            self.loss_G.backward()
        
        return self.loss_G

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
            visual_dict['stenosis_pred'] = torch.sigmoid(self.stenosis_pred_viz)
            visual_dict['stenosis_mask'] = self.stenosis_mask
        
        return visual_dict

    @staticmethod
    def _get_focal_loss():
        """Return focal loss for imbalanced segmentation (optional)."""
        class FocalLoss(torch.nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, pred, target):
                # pred: raw logits
                # target: binary mask (0, 1)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
                p = torch.sigmoid(pred)
                focal_weight = self.alpha * (1 - p).pow(self.gamma)
                return (focal_weight * bce).mean()
        
        return FocalLoss()

