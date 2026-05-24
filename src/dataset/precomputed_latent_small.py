import torch
from train_consts import *
import torch.nn.functional as F

class PreencodedLatentsDataset(torch.utils.data.Dataset):
    """
    Loads a single dataset latents file that was saved by precompute_latents().
    Returns dicts that mirror current batch keys:
      {'image': latent, 'label': y, 'id': sample_id}
    """
    def __init__(self, dataset_path, map_location="cpu", split='train'):
        super().__init__()
        pkg = torch.load(dataset_path, map_location=map_location)
        self.latents = pkg["latents"]          
        self.labels  = pkg["labels"].long()    
        self.ids     = pkg["ids"].long()       
        if 'images' in pkg:
            self.images = pkg['images']        

        

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, i):
        if hasattr(self, 'images'):
            return {
                "image": self.latents[i],   #  NOT pixel space
                "label": self.labels[i],
                "id":    self.ids[i],
                "rgb":   self.images[i],
            }
        else:
         return {"image": self.latents[i],   # already latent, NOT pixel space
                "label": self.labels[i],
                "id":    self.ids[i]
            }        

