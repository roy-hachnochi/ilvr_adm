from PIL import Image
import os

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def check_experiment(exp_name, path, const_axes):
    if not os.path.isdir(os.path.join(path, exp_name)):
        return False
    return (const_axes is None) or (len(const_axes) == 0) or all([name in subdir for name in const_axes])

# ======================================================================================================================
im_dir = os.path.join("ref_imgs", "images_editing")
const_axes = ['images']
size = 256
n_images = 4
# ======================================================================================================================

out_filename = f"summary.png" if (const_axes is None) or len(const_axes) == 0 else f"summary_{'_'.join(const_axes)}.png"
out_filename = os.path.join(im_dir, out_filename)

# load images
im_list = {}
for subdir in os.listdir(im_dir):
    if check_experiment(subdir, im_dir, const_axes):
        files = os.listdir(os.path.join(im_dir, subdir))
        files = list(filter(lambda f: (".png" in f) or (".jpg" in f), files))
        im_list[subdir] = []
        for filename in sorted(files)[:n_images]:
            im_list[subdir].append(Image.open(os.path.join(im_dir, subdir, filename)).resize((size, size)))

# make grid
imgs = []
for i in range(n_images):
    for key in sorted(im_list.keys()):
        imgs.append(im_list[key][i])
grid = image_grid(imgs, n_images, len(im_list))
grid.save(out_filename)

# fig, axes = plt.subplots(n_images, len(im_list))
# for i in range(n_images):
#     for j, key in enumerate(sorted(im_list.keys())):
#         axes[i, j].imshow(im_list[key][i])
#         axes[i, j].set_yticks([])
#         axes[i, j].set_xticks([])
#         if i == n_images - 1:
#             axes[i, j].set_xlabel(key)
# plt.tight_layout()
# plt.suptitle(title)
# plt.savefig(out_filename)
