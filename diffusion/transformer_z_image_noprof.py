from .transformer_z_image import ZImageTransformer2DModel as _ZImageTransformer2DModel


class ZImageTransformer2DModel(_ZImageTransformer2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_internal_profiling(False)
