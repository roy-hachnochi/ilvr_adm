from PIL import Image
import os

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# ======================================================================================================================
parent_dir = os.path.join("output", "TEMP_TIMEMASK_0.1_FILTERED")
im_dir = "refs"
n_images = 4
period = 20
steps = 1000
filename = 'filtered_refs.gif'
# ======================================================================================================================

out_filename = os.path.join(parent_dir, filename)

# load images
im_list = []
for t in range((steps - 1) // period * period, -1, -period):
    imgs = []
    for i in range(n_images):
        filename = os.path.join(parent_dir, im_dir, f"{str(i).zfill(5)}_{str(t).zfill(4)}.png")
        imgs.append(Image.open(filename))
    im_list.append(image_grid(imgs, int((n_images + 1) ** 0.5), int((n_images + 1) ** 0.5)))

im_list[0].save(out_filename, save_all=True, append_images=im_list[1:])
