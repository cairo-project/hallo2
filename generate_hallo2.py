"""
Hallo2: Long-form audio-driven portrait animation.

Public API:
  load_models(...)           -> dict  – load all model weights onto GPU
  generate_video(models, source_image, driving_audio, save_path, **kwargs) -> str
                                      – synthesise an audio-driven video from a
                                        reference portrait image and audio file
  generate(args)             -> str   – convenience: load_models + generate_video

CLI:
  python generate_hallo2.py --source_image face.jpg --driving_audio speech.wav --output out.mp4

Note: importing this module inserts the Hallo2 repo root into sys.path so that
`from hallo.xxx` imports resolve correctly regardless of working directory.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_HALLO2_ROOT = Path(__file__).resolve().parent
if str(_HALLO2_ROOT) not in sys.path:
    sys.path.insert(0, str(_HALLO2_ROOT))

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn

from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.util import tensor_to_video_batch, merge_videos

_DEFAULT_MODEL_DIR = os.environ.get("HALLO2_MODEL_DIR", "/fsx/shared/users/landz/models/hallo2")
_DEFAULT_CONFIG_PATH = str(_HALLO2_ROOT / "configs" / "inference" / "long.yaml")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Net(nn.Module):
    def __init__(self, reference_unet, denoising_unet, face_locator, imageproj, audioproj):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self):
        pass


def _process_audio_emb(audio_emb):
    concatenated_tensors = []
    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)] for j in range(-2, 3)
        ]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
    return torch.stack(concatenated_tensors, dim=0)


def _cut_audio(audio_path, save_dir, length=60):
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(audio_path)
    segment_length = length * 1000
    num_segments = len(audio) // segment_length + (1 if len(audio) % segment_length != 0 else 0)
    os.makedirs(save_dir, exist_ok=True)
    audio_list = []
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, len(audio))
        segment = audio[start_time:end_time]
        path = f"{save_dir}/segment_{i + 1}.wav"
        audio_list.append(path)
        segment.export(path, format="wav")
    return audio_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_models(
    model_dir: str = None,
    config_path: str = None,
    gpu_id: int = 0,
    weight_dtype: str = "fp16",
) -> dict:
    """Load all Hallo2 model weights and return a model bundle dict.

    Args:
        model_dir:    Root directory containing all pretrained models.
                      Must contain subdirectories: stable-diffusion-v1-5,
                      motion_module/mm_sd_v15_v2.ckpt, hallo2/net.pth,
                      face_analysis, wav2vec/wav2vec2-base-960h,
                      audio_separator/Kim_Vocal_2.onnx, sd-vae-ft-mse.
                      Defaults to $HALLO2_MODEL_DIR.
        config_path:  Path to inference YAML config. Defaults to
                      configs/inference/long.yaml next to this file.
        gpu_id:       CUDA device index.
        weight_dtype: "fp16" (default), "bf16", or "fp32".

    Returns:
        dict with keys: pipeline, audioproj, device, weight_dtype_torch,
        img_size, clip_length, n_motion_frames, fps, sample_rate,
        face_analysis_model_path, wav2vec_model_path,
        wav2vec_only_last_features, audio_separator_model_file.
    """
    if model_dir is None:
        model_dir = _DEFAULT_MODEL_DIR
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    if weight_dtype == "fp16":
        wdtype = torch.float16
    elif weight_dtype == "bf16":
        wdtype = torch.bfloat16
    else:
        wdtype = torch.float32

    config = OmegaConf.load(config_path)

    base_model_path = os.path.join(model_dir, "stable-diffusion-v1-5")
    motion_module_path = os.path.join(model_dir, "motion_module", "mm_sd_v15_v2.ckpt")
    audio_ckpt_dir = os.path.join(model_dir, "hallo2")
    vae_model_path = os.path.join(model_dir, "sd-vae-ft-mse")
    face_analysis_model_path = os.path.join(model_dir, "face_analysis")
    wav2vec_model_path = os.path.join(model_dir, "wav2vec", "wav2vec2-base-960h")
    audio_separator_model_file = os.path.join(
        model_dir, "audio_separator", "Kim_Vocal_2.onnx"
    )

    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)

    print(f"Loading VAE from {vae_model_path}")
    vae = AutoencoderKL.from_pretrained(vae_model_path)

    print(f"Loading reference UNet from {base_model_path}")
    reference_unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")

    print(f"Loading denoising UNet from {base_model_path}")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        base_model_path,
        motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs),
        use_landmark=False,
    )

    face_locator = FaceLocator(conditioning_embedding_channels=320)
    image_proj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    )
    audio_proj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device=device, dtype=wdtype)

    for model in (vae, image_proj, reference_unet, denoising_unet, face_locator, audio_proj):
        model.requires_grad_(False)

    reference_unet.enable_gradient_checkpointing()
    denoising_unet.enable_gradient_checkpointing()

    net = _Net(reference_unet, denoising_unet, face_locator, image_proj, audio_proj)

    net_ckpt = os.path.join(audio_ckpt_dir, "net.pth")
    print(f"Loading weights from {net_ckpt}")
    m, u = net.load_state_dict(torch.load(net_ckpt, map_location="cpu"))
    assert len(m) == 0 and len(u) == 0, "Failed to load correct checkpoint."

    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=val_noise_scheduler,
        image_proj=net.imageproj,
    )
    pipeline.to(device=device, dtype=wdtype)

    print("All models loaded.")
    return {
        "pipeline": pipeline,
        "audioproj": net.audioproj,
        "device": device,
        "weight_dtype_torch": wdtype,
        "img_size": (config.data.source_image.width, config.data.source_image.height),
        "clip_length": config.data.n_sample_frames,
        "n_motion_frames": config.data.n_motion_frames,
        "fps": config.data.export_video.fps,
        "sample_rate": config.data.driving_audio.sample_rate,
        "face_analysis_model_path": face_analysis_model_path,
        "wav2vec_model_path": wav2vec_model_path,
        "wav2vec_only_last_features": config.wav2vec.features == "last",
        "audio_separator_model_file": audio_separator_model_file,
    }


@torch.no_grad()
def generate_video(
    models: dict,
    source_image: str,
    driving_audio: str,
    save_path: str,
    *,
    pose_weight: float = 1.0,
    face_weight: float = 1.0,
    lip_weight: float = 1.0,
    face_expand_ratio: float = 1.2,
    inference_steps: int = 40,
    cfg_scale: float = 3.5,
    use_mask: bool = True,
    mask_rate: float = 0.25,
    use_cut: bool = True,
) -> str:
    """Generate a long-form audio-driven portrait video from a source image.

    Args:
        models:           Bundle returned by ``load_models()``.
        source_image:     Path to source portrait image (.jpg/.png).
        driving_audio:    Path to driving audio (.wav, 16 kHz mono).
        save_path:        Output .mp4 path.
        pose_weight:      Motion scale weight for pose.
        face_weight:      Motion scale weight for face.
        lip_weight:       Motion scale weight for lips.
        face_expand_ratio: Face region expansion ratio for masking.
        inference_steps:  Number of DDIM denoising steps.
        cfg_scale:        Classifier-free guidance scale.
        use_mask:         Apply random motion masks during inference.
        mask_rate:        Fraction of pixels masked when use_mask is True.
        use_cut:          Split long audio into 60-second segments
                          before processing (recommended for long audio).

    Returns:
        Absolute path to the saved video (== ``save_path``).
    """
    pipeline = models["pipeline"]
    audioproj = models["audioproj"]
    device = models["device"]
    img_size = models["img_size"]
    clip_length = models["clip_length"]
    n_motion_frames = models["n_motion_frames"]
    fps = models["fps"]
    sample_rate = models["sample_rate"]
    face_analysis_model_path = models["face_analysis_model_path"]
    wav2vec_model_path = models["wav2vec_model_path"]
    wav2vec_only_last_features = models["wav2vec_only_last_features"]
    audio_separator_model_file = models["audio_separator_model_file"]

    motion_scale = [pose_weight, face_weight, lip_weight]
    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_cache_dir = os.path.join(tmpdir, "img_cache")
        seg_video_dir = os.path.join(tmpdir, "seg_video")
        os.makedirs(img_cache_dir)
        os.makedirs(seg_video_dir)

        # --- Process source image ---
        print("Processing source image...")
        with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
            (
                source_image_pixels,
                source_image_face_region,
                source_image_face_emb,
                source_image_full_mask,
                source_image_face_mask,
                source_image_lip_mask,
            ) = image_processor.preprocess(source_image, img_cache_dir, face_expand_ratio)

        # --- Process audio ---
        print("Processing audio...")
        audio_proc_dir = os.path.join(tmpdir, "audio_preprocess")
        if use_cut:
            audio_seg_dir = os.path.join(tmpdir, "audio_segs")
            audio_list = _cut_audio(driving_audio, audio_seg_dir)
            audio_processor = AudioProcessor(
                sample_rate,
                fps,
                wav2vec_model_path,
                wav2vec_only_last_features,
                os.path.dirname(audio_separator_model_file),
                os.path.basename(audio_separator_model_file),
                audio_proc_dir,
            )
            audio_emb_list = []
            processed_length = 0
            for idx, seg_path in enumerate(audio_list):
                padding = (idx + 1) == len(audio_list)
                emb, length = audio_processor.preprocess(
                    seg_path, clip_length, padding=padding, processed_length=processed_length
                )
                audio_emb_list.append(emb)
                processed_length += length
            audio_emb = torch.cat(audio_emb_list)
            audio_length = processed_length
        else:
            with AudioProcessor(
                sample_rate,
                fps,
                wav2vec_model_path,
                wav2vec_only_last_features,
                os.path.dirname(audio_separator_model_file),
                os.path.basename(audio_separator_model_file),
                audio_proc_dir,
            ) as audio_processor:
                audio_emb, audio_length = audio_processor.preprocess(driving_audio, clip_length)

        audio_emb = _process_audio_emb(audio_emb)

        # --- Prepare tensors ---
        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)

        source_image_full_mask = [mask.repeat(clip_length, 1) for mask in source_image_full_mask]
        source_image_face_mask = [mask.repeat(clip_length, 1) for mask in source_image_face_mask]
        source_image_lip_mask = [mask.repeat(clip_length, 1) for mask in source_image_lip_mask]

        times = audio_emb.shape[0] // clip_length
        tensor_result = []
        generator = torch.manual_seed(42)
        batch_size = 60
        start = 0

        # --- Inference loop ---
        for t in range(times):
            print(f"[{t + 1}/{times}]")

            if len(tensor_result) == 0:
                motion_zeros = source_image_pixels.repeat(n_motion_frames, 1, 1, 1)
                motion_zeros = motion_zeros.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device
                )
                pixel_values_ref_img = torch.cat([source_image_pixels, motion_zeros], dim=0)
            else:
                motion_frames = tensor_result[-1][0]
                motion_frames = motion_frames.permute(1, 0, 2, 3)
                motion_frames = motion_frames[0 - n_motion_frames:]
                motion_frames = motion_frames * 2.0 - 1.0
                motion_frames = motion_frames.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device
                )
                pixel_values_ref_img = torch.cat([source_image_pixels, motion_frames], dim=0)

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
            pixel_motion_values = pixel_values_ref_img[:, 1:]

            if use_mask:
                b, f, c, h, w = pixel_motion_values.shape
                rand_mask = torch.rand(h, w)
                mask = (rand_mask > mask_rate).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, f, c, h, w)
                face_mask_t = source_image_face_region.repeat(f, 1, 1, 1).unsqueeze(0)
                mask = mask | face_mask_t.bool()
                pixel_motion_values = pixel_motion_values * mask
                pixel_values_ref_img[:, 1:] = pixel_motion_values

            audio_tensor = audio_emb[t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])]
            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(device=audioproj.device, dtype=audioproj.dtype)
            audio_tensor = audioproj(audio_tensor)

            pipeline_output = pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=source_image_face_emb,
                face_mask=source_image_face_region,
                pixel_values_full_mask=source_image_full_mask,
                pixel_values_face_mask=source_image_face_mask,
                pixel_values_lip_mask=source_image_lip_mask,
                width=img_size[0],
                height=img_size[1],
                video_length=clip_length,
                num_inference_steps=inference_steps,
                guidance_scale=cfg_scale,
                generator=generator,
                motion_scale=motion_scale,
            )
            tensor_result.append(pipeline_output.videos)

            if (t + 1) % batch_size == 0 or (t + 1) == times:
                last_motion_frame = [tensor_result[-1]]

                if start != 0:
                    batch_tensor = torch.cat(tensor_result[1:], dim=2)
                else:
                    batch_tensor = torch.cat(tensor_result, dim=2)

                batch_tensor = batch_tensor.squeeze(0)
                f = batch_tensor.shape[1]
                length = min(f, audio_length)
                batch_tensor = batch_tensor[:, :length]

                seg_output = os.path.join(seg_video_dir, f"segment-{t + 1:06}.mp4")
                tensor_to_video_batch(batch_tensor, seg_output, start, driving_audio, fps=fps)
                del batch_tensor

                tensor_result = last_motion_frame
                audio_length -= length
                start += length

        # --- Merge segments into final output ---
        print("Merging video segments...")
        merge_videos(seg_video_dir, save_path)

    print(f"Saved to {save_path}")
    return os.path.abspath(save_path)


def generate(args) -> str:
    """Convenience wrapper: load models then generate one video."""
    models = load_models(
        model_dir=args.model_dir,
        config_path=args.config_path,
        gpu_id=args.gpu_id,
        weight_dtype=args.weight_dtype,
    )
    return generate_video(
        models=models,
        source_image=args.source_image,
        driving_audio=args.driving_audio,
        save_path=args.output,
        pose_weight=args.pose_weight,
        face_weight=args.face_weight,
        lip_weight=args.lip_weight,
        face_expand_ratio=args.face_expand_ratio,
        inference_steps=args.inference_steps,
        cfg_scale=args.cfg_scale,
        use_mask=args.use_mask,
        mask_rate=args.mask_rate,
        use_cut=args.use_cut,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hallo2 long-form audio-driven portrait inference")

    parser.add_argument("--source_image", type=str, required=True,
                        help="Source portrait image (.jpg/.png)")
    parser.add_argument("--driving_audio", type=str, required=True,
                        help="Driving audio file (.wav, 16 kHz mono)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output video path (.mp4)")
    parser.add_argument("--model_dir", type=str, default=_DEFAULT_MODEL_DIR,
                        help="Root directory containing pretrained models")
    parser.add_argument("--config_path", type=str, default=_DEFAULT_CONFIG_PATH,
                        help="Path to inference YAML config")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA device index")
    parser.add_argument("--weight_dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"],
                        help="Model weight dtype")
    parser.add_argument("--pose_weight", type=float, default=1.0,
                        help="Motion scale weight for pose")
    parser.add_argument("--face_weight", type=float, default=1.0,
                        help="Motion scale weight for face")
    parser.add_argument("--lip_weight", type=float, default=1.0,
                        help="Motion scale weight for lips")
    parser.add_argument("--face_expand_ratio", type=float, default=1.2,
                        help="Face region expansion ratio")
    parser.add_argument("--inference_steps", type=int, default=40,
                        help="Number of DDIM denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=3.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--use_mask", action="store_true", default=True,
                        help="Apply random motion masks (default: on)")
    parser.add_argument("--no_mask", dest="use_mask", action="store_false",
                        help="Disable motion masks")
    parser.add_argument("--mask_rate", type=float, default=0.25,
                        help="Fraction of pixels masked when --use_mask is set")
    parser.add_argument("--use_cut", action="store_true", default=True,
                        help="Split audio into 60-second segments (default: on)")
    parser.add_argument("--no_cut", dest="use_cut", action="store_false",
                        help="Disable audio segmentation")

    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
