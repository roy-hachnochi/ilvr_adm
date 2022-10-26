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
im_dir = os.path.join("output", "imagenet", "TEST_WithRePaint")
n_images = 4
# ======================================================================================================================

out_filename = os.path.join(im_dir, "summary.png")

# load images
im_list = {}
for subdir in os.listdir(im_dir):
    if os.path.isdir(os.path.join(im_dir, subdir)):
        files = os.listdir(os.path.join(im_dir, subdir))
        files = list(filter(lambda f: ".png" in f, files))
        im_list[subdir] = []
        for filename in sorted(files)[:n_images]:
            im_list[subdir].append(Image.open(os.path.join(im_dir, subdir, filename)))

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
