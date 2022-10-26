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
    return (const_axes is None) or (len(const_axes) == 0) or all([name + "_" in subdir for name in const_axes])

# ======================================================================================================================
im_dir = os.path.join("output", "imagenet_inpaint_maskt_blurin_blend")
row_axis = 'blursigmain'
col_axis = 'timemask'
const_axes = ['blendpix_30']
im_i = 0
# ======================================================================================================================

out_filename = f"summary_{im_i}.png" if (const_axes is None) or len(const_axes) == 0 else f"summary_{'_'.join(const_axes)}_{im_i}.png"
out_filename = os.path.join(im_dir, out_filename)

# load images
im_list = {}
for subdir in os.listdir(im_dir):
    if check_experiment(subdir, im_dir, const_axes):
        files = os.listdir(os.path.join(im_dir, subdir))
        files = list(filter(lambda f: ".png" in f, files))
        x = float(subdir.split(row_axis)[1].split('_')[1])
        y = float(subdir.split(col_axis)[1].split('_')[1])
        if x not in im_list:
            im_list[x] = {}
        filename = sorted(files)[im_i]
        im_list[x][y] = Image.open(os.path.join(im_dir, subdir, filename))

# make grid
imgs = []
for x in sorted(im_list.keys()):  # rows
    for y in sorted(im_list[x].keys()):  # cols
        imgs.append(im_list[x][y])
grid = image_grid(imgs, len(im_list), len(im_list[list(im_list.keys())[0]]))
grid.save(out_filename)
