__all__ = ["CrossImageStego", "ImageStegoResult"]


def __getattr__(name):
    if name in {"CrossImageStego", "ImageStegoResult"}:
        from stegox.image.cross import CrossImageStego, ImageStegoResult

        return {"CrossImageStego": CrossImageStego, "ImageStegoResult": ImageStegoResult}[name]
    raise AttributeError(name)
