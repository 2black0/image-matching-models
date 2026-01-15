import torch
import torch.nn as nn

def patch_romav2_matcher():
    """
    Monkey patches RoMaV2 Matcher.__init__ to use register_buffer instead of nn.Buffer.
    """
    try:
        from romav2.matcher import Matcher
        from romav2.types import MatcherStyle
        from romav2.vit import vit_from_name
        from romav2.dpt import DPTHead
        from romav2.device import device
    except ImportError:
        return

    original_init = Matcher.__init__


    def new_init(self, cfg):
        # We cannot call super().__init__() because we are replacing the whole init logic 
        # to avoid the specific lines that fail.
        # Alternatively, we can copy the code.
        # Mirroring the fixed code:
        nn.Module.__init__(self)
        self.cfg = cfg
        omega = 2 * torch.pi * torch.randn(cfg.dim // 2, 2)
        # Fix: use register_buffer
        self.register_buffer("omega", omega)
        self.register_buffer("scale", torch.tensor(cfg.scale))
        self.register_buffer("temp", torch.tensor(cfg.temp))
        
        self.mv_vit = vit_from_name(
            cfg.mv_vit,
            device=device,
            in_dim=cfg.feat_dim * cfg.num_feature_layers,
            out_dim=cfg.dim,
            multiview=True,
            use_rope=cfg.mv_vit_use_rope,
            mv_position_mode=cfg.mv_vit_position_mode,
            mv_attention_mode=cfg.mv_vit_attention_mode,
            pos_embed_rope_rescale_coords=cfg.pos_embed_rope_rescale_coords,
        )
        self.head = DPTHead(
            dim_in=cfg.dim,
            out_dim=cfg.warp_dim + cfg.confidence_dim,
            pos_embed=False,
            feature_only=False,
            down_ratio=4,
        )

    Matcher.__init__ = new_init
    print("Monkey-patched RoMaV2 Matcher.__init__ to fix nn.Buffer usage.")
