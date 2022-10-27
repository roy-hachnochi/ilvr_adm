import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import sys
from omegaconf import OmegaConf

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
import math


# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for model_kwargs in data:
        yield model_kwargs


def main():
    parser, defaults = create_argparser()
    args, unknown = parser.parse_known_args()
    default_conf = OmegaConf.create({'sampling': args_to_dict(args, defaults['sampling'].keys()),
                                     'model_and_diffusion': args_to_dict(args, defaults['model_and_diffusion'].keys())})
    cli = OmegaConf.from_dotlist(unknown)
    configs = [OmegaConf.load(cfg) for cfg in args.config]
    config = OmegaConf.merge(default_conf, *configs, cli)
    sampling_conf = config.sampling
    model_and_diffusion_conf = config.model_and_diffusion

    # TODO: move this to parser?
    sampling_conf.mask = float(sampling_conf.mask) if sampling_conf.mask else None
    sampling_conf.T_mask = float(sampling_conf.T_mask) if sampling_conf.T_mask else None
    sampling_conf.down_N_in = int(sampling_conf.down_N_in) if sampling_conf.down_N_in else None
    sampling_conf.down_N_out = int(sampling_conf.down_N_out) if sampling_conf.down_N_out else None
    sampling_conf.fft = int(sampling_conf.fft) if sampling_conf.fft else None
    sampling_conf.blend_pix = int(sampling_conf.blend_pix) if sampling_conf.blend_pix else None
    sampling_conf.blur_sigma_in = float(sampling_conf.blur_sigma_in) if sampling_conf.blur_sigma_in else None
    sampling_conf.blur_sigma_out = float(sampling_conf.blur_sigma_out) if sampling_conf.blur_sigma_out else None

    if sampling_conf.seed >= 0:
        th.manual_seed(sampling_conf.seed)
        np.random.seed(sampling_conf.seed)

    dist_util.setup_dist()
    logger.configure(dir=sampling_conf.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_conf)
    model.load_state_dict(dist_util.load_state_dict(model_and_diffusion_conf.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if model_and_diffusion_conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    assert ((sampling_conf.down_N_in is None) or math.log(sampling_conf.down_N_in, 2).is_integer())
    assert ((sampling_conf.down_N_out is None) or math.log(sampling_conf.down_N_out, 2).is_integer())

    logger.log("loading data...")
    data = load_reference(
        sampling_conf.base_samples,
        sampling_conf.batch_size,
        image_size=model_and_diffusion_conf.image_size,
        class_cond=model_and_diffusion_conf.class_cond,
    )

    if sampling_conf.save_latents > 0:
        save_latents = {'dir': os.path.join(logger.get_dir(), "latents"), 'period': sampling_conf.save_latents}
        os.makedirs(save_latents['dir'], exist_ok=True)
    else:
        save_latents = None
    if sampling_conf.save_refs > 0:
        save_refs = {'dir': os.path.join(logger.get_dir(), "refs"), 'period': sampling_conf.save_refs}
        os.makedirs(save_refs['dir'], exist_ok=True)
    else:
        save_refs = None

    with open(os.path.join(logger.get_dir(), 'run_command'), 'w') as f:
        f.write(" ".join(f"'{i}'" if " " in i else i for i in sys.argv))
    OmegaConf.save(config, os.path.join(logger.get_dir(), 'config.yaml'))

    logger.log("creating samples...")
    count = 0
    while count * sampling_conf.batch_size < sampling_conf.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(model, (
            sampling_conf.batch_size, 3, model_and_diffusion_conf.image_size, model_and_diffusion_conf.image_size),
                                         clip_denoised=sampling_conf.clip_denoised,
                                         model_kwargs=model_kwargs,
                                         down_N_out=sampling_conf.down_N_out,
                                         range_t=sampling_conf.range_t,
                                         blend_pix=sampling_conf.blend_pix,
                                         mask_alpha=sampling_conf.mask,
                                         T_mask=sampling_conf.T_mask,
                                         down_N_in=sampling_conf.down_N_in,
                                         fft_num=sampling_conf.fft,
                                         blur_sigma_in=sampling_conf.blur_sigma_in,
                                         blur_sigma_out=sampling_conf.blur_sigma_out,
                                         save_latents=save_latents,
                                         save_refs=save_refs)

        for i in range(sampling_conf.batch_size):
            out_path = os.path.join(logger.get_dir(),
                                    f"{str(count * sampling_conf.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * sampling_conf.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    sample_defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        down_N_in="",
        range_t=0.,
        use_ddim=False,
        base_samples="",
        save_dir="",
        save_latents=-1,
        save_refs=-1,
        seed=-1,
        blend_pix="",
        mask="",
        T_mask="",
        down_N_out="",
        fft="",
        blur_sigma_in="",
        blur_sigma_out="",
    )
    model_defaults = model_and_diffusion_defaults()
    model_defaults['model_path'] = ""
    defaults = {'sampling': sample_defaults,
                'model_and_diffusion': model_defaults}
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, sample_defaults)
    add_dict_to_argparser(parser, model_defaults)
    parser.add_argument(
        "--config",
        nargs="*",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser, defaults


if __name__ == "__main__":
    main()
