import torch
import torch.nn.functional as F

from dloc.core.utils.base_model import BaseModel  # noqa: E402
from src.build_samatcher import build_samatcher


class SAMatcher(BaseModel):
    default_conf = {
        'weights': 'samatcher_best.ckpt',
        'sam_cfg': 'configs/sam2.1/sam2.1_hiera_l_sammatcher.yaml',
        'sam_checkpoint': 'checkpoints/sam2.1_hq_hiera_large.pt',
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def _init(self, conf, model_path):
        # pdb.set_trace()
        self.conf = {**self.default_conf, **conf}
        
        self.model = build_samatcher(
            self.conf['sam_cfg'], 
            self.conf['sam_checkpoint']
        ).eval()
        
        # Load the trained weights if available
        model_file = model_path / self.conf['weights']
        if model_file.exists():
            state_dict = torch.load(model_file, map_location='cpu')
            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)

    def _forward(self, data):
        # Extract original image dimensions
        h0, w0 = data['image0'].shape[1:3]  # height, width of image0
        h1, w1 = data['image1'].shape[1:3]  # height, width of image1
        
        masks, boxes = self.model.forward(data['image0'].permute(0, 3, 1, 2), data['image1'].permute(0, 3, 1, 2))
        bbox0, bbox1 = boxes
        mask0_o, mask1_o = masks[0, 0], masks[0, 1]
        
        # Reshape masks to original image size
        mask0 = F.interpolate(mask0_o.unsqueeze(0).unsqueeze(0), size=(h0, w0), mode='bilinear', align_corners=False).squeeze()
        mask1 = F.interpolate(mask1_o.unsqueeze(0).unsqueeze(0), size=(h1, w1), mode='bilinear', align_corners=False).squeeze()
        
        return bbox0, bbox1, mask0, mask1, mask0_o.unsqueeze(0).unsqueeze(0), mask1_o.unsqueeze(0).unsqueeze(0)
