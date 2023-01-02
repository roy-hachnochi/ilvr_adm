from PIL import Image
import os
from tqdm import tqdm

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# ======================================================================================================================
parent_dir = os.path.join("output", "imagenet", "TEST2_yes", "downNin_1_ranget_0.8_seed_42_blendpix_30_Tmask_0.8_exp_000")
im_dir = "latents"
n_images = 4
period = 2
start_step = 3500
end_step = 4570
filename = 'latents_short.gif'
# ======================================================================================================================

out_filename = os.path.join(parent_dir, filename)

# load images
im_list = []
for t in tqdm(range(start_step, end_step, period)):
    imgs = []
    for i in range(n_images):
        filename = os.path.join(parent_dir, im_dir, f"{str(i).zfill(5)}_{str(t).zfill(4)}.png")
        imgs.append(Image.open(filename))
    im_list.append(image_grid(imgs, int((n_images + 1) ** 0.5), int((n_images + 1) ** 0.5)))

im_list[0].save(out_filename, save_all=True, append_images=im_list[1:])
