from __future__ import annotations

import torch


def make_ddim_scheduler():
    from diffusers import DDIMScheduler

    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )


def load_stable_diffusion(
    model_name: str = "runwayml/stable-diffusion-v1-5",
    device: str | None = None,
    dtype: torch.dtype | None = None,
):
    from diffusers import StableDiffusionPipeline

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None and resolved_device.startswith("cuda"):
        dtype = torch.float16

    kwargs = {"scheduler": make_ddim_scheduler()}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    pipe = StableDiffusionPipeline.from_pretrained(model_name, **kwargs)
    pipe = pipe.to(resolved_device)
    try:
        pipe.disable_xformers_memory_efficient_attention()
    except AttributeError:
        pass
    pipe.set_progress_bar_config(disable=True)
    return pipe
