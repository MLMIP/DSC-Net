from .model1 import Swin
def get_seg_model(cfg, **kwargs):
    model = Swin()
    return model