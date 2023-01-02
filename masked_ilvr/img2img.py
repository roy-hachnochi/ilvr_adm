"""make variations of input image"""

import argparse, os, sys
from itertools import product
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf, ListConfig
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from torchvision.utils import make_grid
from scipy.ndimage import binary_dilation
import torch.distributed as dist

from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init_img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--use_ddpm",
        action='store_true',
        help="use DDPM instead of DDIM for sampling.",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        nargs='*',
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given image",
    )
    parser.add_argument(
        "--max_imgs",
        type=int,
        nargs='?',
        default=1000000,
        help="max number of images from input",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size (number of images in single forward pass)",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--strength",
        type=float,
        nargs='*',
        default=1.0,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/royha/ilvr_adm/configs/imagenet_512_uncond_model.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/disk2/royha/guided-diffusion/512x512_diffusion_uncond_finetune_008100.pt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs='*',
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--mask",
        type=str,
        nargs="?",
        help="path to mask"
    )
    parser.add_argument(
        "--down_N_in",
        type=int,
        nargs='*',
        help="ILVR downsampling factor inside mask"
    )
    parser.add_argument(
        "--down_N_out",
        type=int,
        nargs='*',
        help="ILVR downsampling factor outside mask"
    )
    parser.add_argument(
        "--T_out",
        type=float,
        nargs='*',
        default=0.0,
        help="strength of ILVR outside mask (in [0.0, 0.1])"
    )
    parser.add_argument(
        "--T_in",
        type=float,
        nargs='*',
        default=0.0,
        help="strength of ILVR inside mask (in [0.0, 0.1])"
    )
    parser.add_argument(
        "--blend_pix",
        type=int,
        nargs='*',
        help="number of pixels for mask smoothing"
    )
    parser.add_argument(
        "--repaint_start",
        type=float,
        nargs='*',
        default=0.0,
        help="Use RePaint (https://arxiv.org/pdf/2201.09865.pdf) for conditioning for (r*time_steps steps), (r in [0.0, 0.1])",
    )
    parser.add_argument(
        "--mask_dilate",
        type=int,
        nargs='*',
        default=0,
        help="Dilate mask to contain larger region (# pixels to dilate)",
    )
    parser.add_argument(
        "--shadow",
        action='store_true',
        help="adjust mask for shadow generation",
    )

    return parser

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((512, 512), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def load_mask(path, size, dilate, shadow=False):
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, resample=PIL.Image.Resampling.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    if shadow:
        mask = add_shadow(mask)
    mask = mask[None, None]
    mask[mask < 0.05] = 0
    mask[mask >= 0.05] = 1
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate).astype(np.float32)
    mask = torch.from_numpy(mask)
    return mask

def add_shadow(mask, width=30):
    shadow_mask = np.zeros_like(mask)
    for object_val in np.unique(mask):
        if object_val == 0:
            continue
        u, d = np.where(np.any(mask == object_val, axis=1))[0][[0, -1]]
        l, r = np.where(np.any(mask == object_val, axis=0))[0][[0, -1]]
        u, d = max(d - width, 0), min(d + width, mask.shape[0])
        l, r = max(l - width, 0), min(r + width, mask.shape[1])
        # mask[u:d, l:r] = np.where(mask[u:d, l:r] == 0, object_val, mask[u:d, l:r])
        shadow_mask[u:d, l:r] = object_val
    return shadow_mask

def load_data(img_path, mask_path, max_imgs, batch_size, mask_dilate, shadow=False):
    print(f"reading images from {img_path}")
    im_list = [os.path.join(img_path, filename) for filename in sorted(os.listdir(img_path))] if os.path.isdir(
        img_path) else [img_path]
    n_imgs = len(im_list)
    if mask_path is not None:
        print(f"reading masks from {mask_path}")
        mask_list = [os.path.join(mask_path, filename) for filename in sorted(os.listdir(mask_path))] if os.path.isdir(
            mask_path) else [mask_path]
        assert(len(mask_list) == n_imgs)

    img = []
    mask = []
    filename = []
    b = 0
    for i in range(1, min(n_imgs, max_imgs) + 1):
        filename.append(os.path.basename(im_list[i - 1]).split(".")[0])
        img.append(load_img(im_list[i - 1]))
        if mask_path is not None:
            mask.append(load_mask(mask_list[i - 1], tuple(img[-1].shape[-2:][::-1]), mask_dilate, shadow))
        if i % batch_size == 0 or i == n_imgs:
            img = torch.concat(img, dim=0)
            mask = torch.concat(mask, dim=0) if mask_path is not None else None
            yield img, mask, filename
            img, mask, filename = [], [], []
            b += 1

def main():
    # parse arguments
    parser = create_argparser()
    opt, unknown = parser.parse_known_args()
    sampling_conf = OmegaConf.create({'sampling': vars(opt)})
    cli = OmegaConf.from_dotlist(unknown)
    cli = {key.replace('--', ''): val for key, val in cli.items()}
    config = OmegaConf.load(f"{opt.config}")
    config = OmegaConf.merge(sampling_conf, config, cli)
    sampling_conf = config.sampling
    model_and_diffusion_conf = config.model_and_diffusion

    sweep = {}
    for k in iter(sampling_conf):
        if isinstance(sampling_conf[k], ListConfig):
            if len(sampling_conf[k]) > 1:
                sweep[k] = [(k, x) for x in sampling_conf[k]]
            else:
                sampling_conf[k] = sampling_conf[k][0]

    model_and_diffusion_conf.timestep_respacing = f'{sampling_conf.steps}' if sampling_conf.use_ddpm else f'ddim{sampling_conf.steps}'

    # load model and sampler
    dist_util.setup_dist()
    device = dist_util.dev()
    model, sampler = create_model_and_diffusion(**model_and_diffusion_conf)
    model.load_state_dict(dist_util.load_state_dict(sampling_conf.ckpt, map_location="cpu"))
    model.to(device)
    if model_and_diffusion_conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    sampling_fn = sampler.p_sample_loop if sampling_conf.use_ddpm else sampler.ddim_sample_loop

    nrow = sampling_conf.nrow if sampling_conf.nrow > 0 else sampling_conf.n_samples

    os.makedirs(sampling_conf.outdir, exist_ok=True)
    outpath = sampling_conf.outdir
    with open(os.path.join(outpath, 'run_command'), 'w') as f:
        f.write(" ".join(f"\"{i}\"" if " " in i else i for i in sys.argv))
    OmegaConf.save(config, os.path.join(outpath, 'config.yaml'))

    print('Sweeping through: {}'.format({key: [val[1] for val in sweep[key]] for key in sweep}))
    for exp_i, prod in enumerate(product(*sweep.values())):
        print("===================================================")
        print(f'Running with {prod}')
        for param, val in prod:
            sampling_conf[param] = val

        data_loader = load_data(sampling_conf.init_img, sampling_conf.mask, sampling_conf.max_imgs,
                                sampling_conf.batch_size, sampling_conf.mask_dilate, sampling_conf.shadow)

        sample_path = os.path.join(outpath, '_'.join([str(val).replace('_', '') for element in prod for val in element]))
        os.makedirs(sample_path, exist_ok=True)
        OmegaConf.save(config, os.path.join(sample_path, 'config.yaml'))

        assert 0. <= sampling_conf.strength <= 1., 'can only work with strength in [0.0, 1.0]'
        repaint_conf = OmegaConf.create({'use_repaint': sampling_conf.repaint_start > 0,
                                         'inpa_inj_time_shift': 1,
                                         'schedule_jump_params': {
                                             't_T': sampling_conf.steps,
                                             'n_sample': 1,
                                             'jump_length': 10,
                                             'jump_n_sample': 10,
                                             'start_resampling': int(sampling_conf.repaint_start * sampling_conf.steps),
                                             'collapse_increasing': False}})
        t_enc = int(sampling_conf.strength * sampling_conf.steps)
        print(f"target t_enc is {t_enc} steps")

        grid_count = 0
        with torch.no_grad():
            all_samples = list()
            for img, mask, filename in tqdm(data_loader, desc="Data"):
                if (sampling_conf.seed is not None) and (sampling_conf.seed >= 0):
                    torch.manual_seed(sampling_conf.seed)
                    np.random.seed(sampling_conf.seed)
                img = img.to(device)
                mask = mask.to(device) if mask is not None else None
                batch_size = img.shape[0]

                for n in trange(sampling_conf.n_samples, desc="Sampling"):
                    # encode (scaled latent)
                    if t_enc < sampling_conf.steps:
                        ref_sample = sampler.q_sample(img, torch.tensor([t_enc]*batch_size).to(device))
                    else:  # strength >= 1 ==> use only noise
                        ref_sample = torch.randn_like(img)
                    # decode it
                    samples = sampling_fn(model,
                                          ref_sample.shape,
                                          noise=ref_sample,
                                          device=device,
                                          progress=True,
                                          eta=sampling_conf.ddim_eta,
                                          ref_image=img,
                                          mask=mask,
                                          down_N_out=sampling_conf.down_N_out,
                                          down_N_in=sampling_conf.down_N_in,
                                          T_out=sampling_conf.T_out,
                                          T_in=sampling_conf.T_in,
                                          blend_pix=sampling_conf.blend_pix,
                                          t_start=t_enc,
                                          repaint=repaint_conf)

                    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
                    if not sampling_conf.skip_save:
                        for i, x_sample in enumerate(samples):
                            x_sample = 255. * x_sample.permute(1, 2, 0).cpu().numpy()
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, f"{filename[i]}_{n:02}.png"))
                    all_samples.append(samples)

            if not sampling_conf.skip_grid:
                # additionally, save as grid
                grid = torch.concat(all_samples, 0)
                grid = make_grid(grid, nrow=nrow)

                # to image
                grid = 255. * grid.permute(1, 2, 0).cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(sample_path, f'grid-{grid_count:04}.png'))
                grid_count += 1

            print(f"finished {exp_i + 1}/{np.prod([len(l) for l in sweep.values()])} experiments")

    dist.barrier()
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
