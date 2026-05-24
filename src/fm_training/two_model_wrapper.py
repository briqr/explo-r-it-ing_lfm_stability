import torch
import torch.nn as nn

class CoarseFineModel(nn.Module):
    """
    Wraps two trained models and returns a single forward(x, t, **kw).
    """
    def __init__(self, coarse, fine, transport, device=None,
                 t_split=None,                 ## e.g t_split=0.8: use coarse for t <= 0.8, then switch to fine for t > 0.8
                 ):
        super().__init__()
        self.coarse = coarse.eval()
        self.fine   = fine.eval()
        self.transport = transport
        self.t_split = t_split

            

    @torch.no_grad()
    def forward(self, x, t, **kw):
                
        use_fine = (t.mean() >= self.t_split)
        m = self.fine if use_fine else self.coarse
        return m(x, t, **kw)

