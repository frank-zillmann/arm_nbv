from abc import ABC


class BaseReconstructionPolicy(ABC):
    def __init__(self, **kwargs):
        pass

    def add_obs(
        self,
        camera_intrinsic,
        camera_extrinsic,
        rgb_image=None,
        depth_image=None,
        **kwargs,
    ):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reconstruct(self, type=None, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
