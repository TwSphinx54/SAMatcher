import torch

from dloc.core.utils.base_model import BaseModel  # noqa: E402
from src.config.default import get_cfg_defaults
from src.model import build_detectors


class CCOE(BaseModel):
    default_conf = {
        'model': 'ccoe',
        'num_layers': 50,
        'stride': 32,
        'last_layer': 1024,
        # 'weights': 'detmatcher.pth',
        'weights': 'scode.pth',
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def build_cfg(self, conf):
        cfg = get_cfg_defaults()
        cfg.CCOE.MODEL = conf['model']
        cfg.CCOE.BACKBONE.STRIDE = conf['stride']
        cfg.CCOE.BACKBONE.LAYER = conf['layer']
        cfg.CCOE.BACKBONE.LAST_LAYER = conf['last_layer']
        # cfg.DATASET.TRAIN.IMAGE_SIZE = [480, 480]
        # cfg.DATASET.VAL.IMAGE_SIZE = [480, 480]
        cfg.DATASET.TRAIN.IMAGE_SIZE = [640, 640]
        cfg.DATASET.VAL.IMAGE_SIZE = [640, 640]
        cfg.CCOE.CCA.FEAT_SIZE = cfg.DATASET.TRAIN.IMAGE_SIZE[0] // (2 ** 5)
        cfg.CCOE.CCA.FEAT_CHAN = cfg.CCOE.BACKBONE.LAST_LAYER // 4
        cfg.CCOE.CCA.DEPTH = [2, 2, 2, 2]
        cfg.CCOE.CCA.NUM_HEADS = [8, 8, 8, 8]
        cfg.CCOE.CCA.MSA_SIZES = [[3, 5, 7], [3, 5, 7], [3, 5, 7]]
        cfg.CCOE.CCA.NON_OVERLAP_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]
        return cfg

    def _init(self, conf, model_path):
        # pdb.set_trace()
        self.conf = {**self.default_conf, **conf}
        self.cfg = self.build_cfg(self.conf)
        self.net = build_detectors(self.cfg.CCOE)
        model_file = model_path / self.conf['weights']
        self.net.load_state_dict(torch.load(model_file))

    def _forward(self, data):
        box1, box2, sim = self.net.forward_dummy(data['image0'], data['image1'])
        return box1, box2
