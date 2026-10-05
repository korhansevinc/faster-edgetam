# The EdgeTAM checkpoint overwrites the RepViT weights, so skip timm's ImageNet download.
import sam2.modeling.backbones.timm as _tb

_orig = _tb.create_model


def _create_model_offline(*args, **kwargs):
    kwargs["pretrained"] = False
    return _orig(*args, **kwargs)


_tb.create_model = _create_model_offline
