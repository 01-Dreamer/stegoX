from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from stegox.image.io import load_rgb_image, save_rgb_image
from stegox.image.pipeline import load_stable_diffusion, make_ddim_scheduler


GUIDANCE_SCALE = 1.0


@dataclass
class ImageStegoResult:
    cover_image: np.ndarray | None = None
    stego_image: np.ndarray | None = None
    recovered_image: np.ndarray | None = None


class DDIMODESolver:
    def __init__(self, model, num_steps: int = 50, guidance_scale: float = GUIDANCE_SCALE, show_progress: bool = True):
        self.model = model
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.show_progress = show_progress
        self.model.scheduler = make_ddim_scheduler()
        self.model.scheduler.set_timesteps(num_steps)
        self.context = None
        self.prompt = None

    @property
    def scheduler(self):
        return self.model.scheduler

    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        return alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps,
            999,
        ), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        return alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction

    def noise_pred_step(self, latents, timestep, is_forward: bool):
        if self.context is None:
            raise RuntimeError("prompt context is not initialized")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, timestep, uncond_embeddings)["sample"]
        noise_pred_text = self.model.unet(latents, timestep, cond_embeddings)["sample"]
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        if is_forward:
            return self.next_step(noise_pred, timestep, latents)
        return self.prev_step(noise_pred, timestep, latents)

    @torch.no_grad()
    def latent_to_image(self, latents) -> np.ndarray:
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)["sample"]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        return (image * 255).astype(np.uint8)

    @torch.no_grad()
    def image_to_latent(self, image: np.ndarray):
        vae_dtype = next(self.model.vae.parameters()).dtype
        tensor = torch.from_numpy(image).float() / 127.5 - 1
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.model.device, dtype=vae_dtype)
        latents = self.model.vae.encode(tensor)["latent_dist"].mean
        return latents * 0.18215

    @torch.no_grad()
    def init_prompt(self, prompt: str) -> None:
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    def _steps(self) -> Iterable[int]:
        steps = range(self.num_steps)
        if not self.show_progress:
            return steps
        from tqdm import tqdm

        return tqdm(steps)

    @torch.no_grad()
    def ddim_loop(self, latent, is_forward: bool):
        latent = latent.clone().detach()
        for index in self._steps():
            if is_forward:
                timestep = self.scheduler.timesteps[len(self.scheduler.timesteps) - index - 1]
            else:
                timestep = self.scheduler.timesteps[index]
            latent = self.noise_pred_step(latent, timestep, is_forward)
        return latent

    def invert(self, prompt: str, start_latent, is_forward: bool):
        self.init_prompt(prompt)
        return self.ddim_loop(start_latent, is_forward=is_forward)


class CrossImageStego:
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        num_steps: int = 50,
        device: str | None = None,
        guidance_scale: float = GUIDANCE_SCALE,
        show_progress: bool = True,
    ):
        self.pipeline = load_stable_diffusion(model_name=model_name, device=device)
        self.solver = DDIMODESolver(
            self.pipeline,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            show_progress=show_progress,
        )

    def hide_array(self, secret_image: np.ndarray, private_key: str, public_key: str) -> np.ndarray:
        secret_latent = self.solver.image_to_latent(secret_image)
        noise_latent = self.solver.invert(private_key, secret_latent, is_forward=True)
        stego_latent = self.solver.invert(public_key, noise_latent, is_forward=False)
        return self.solver.latent_to_image(stego_latent)

    def reveal_array(self, stego_image: np.ndarray, private_key: str, public_key: str) -> np.ndarray:
        stego_latent = self.solver.image_to_latent(stego_image)
        noise_latent = self.solver.invert(public_key, stego_latent, is_forward=True)
        recovered_latent = self.solver.invert(private_key, noise_latent, is_forward=False)
        return self.solver.latent_to_image(recovered_latent)

    def hide_file(
        self,
        image_path: str | Path,
        private_key: str,
        public_key: str,
        output_path: str | Path,
        resize: bool = True,
    ) -> np.ndarray:
        secret_image = load_rgb_image(image_path, resize=resize)
        stego_image = self.hide_array(secret_image, private_key, public_key)
        save_rgb_image(stego_image, output_path)
        return stego_image

    def reveal_file(
        self,
        image_path: str | Path,
        private_key: str,
        public_key: str,
        output_path: str | Path,
        resize: bool = True,
    ) -> np.ndarray:
        stego_image = load_rgb_image(image_path, resize=resize)
        recovered_image = self.reveal_array(stego_image, private_key, public_key)
        save_rgb_image(recovered_image, output_path)
        return recovered_image

    def roundtrip(
        self,
        image_path: str | Path,
        private_key: str,
        public_key: str,
        output_dir: str | Path,
        resize: bool = True,
    ) -> ImageStegoResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        secret_image = load_rgb_image(image_path, resize=resize)
        save_rgb_image(secret_image, output_dir / "gt.png")
        stego_image = self.hide_array(secret_image, private_key, public_key)
        save_rgb_image(stego_image, output_dir / "hide.png")
        recovered_image = self.reveal_array(stego_image, private_key, public_key)
        save_rgb_image(recovered_image, output_dir / "reverse.png")
        return ImageStegoResult(
            cover_image=secret_image,
            stego_image=stego_image,
            recovered_image=recovered_image,
        )
