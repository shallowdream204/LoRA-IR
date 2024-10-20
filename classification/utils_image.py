from PIL import Image
import torch

def extract_patches(image, patch_size, overlap_ratio):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches
def process_highres_image(image, processor, select_size):
    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)
