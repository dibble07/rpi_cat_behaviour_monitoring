import torch


def get_best_device() -> torch.device:
    """Identify the best available PyTorch device"""
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        out = torch.device("cuda")

    # Check for Mac GPU (Metal Performance Shaders)
    elif torch.backends.mps.is_available():
        out = torch.device("mps")

    # Fallback to CPU
    else:
        out = torch.device("cpu")

    return out


def expand_bbox_from_bounds(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    image_width: int,
    image_height: int,
    pad: int,
) -> list[int]:
    """Expand a bbox with padding and enforce frame aspect ratio."""

    # identify initial padded bounding box
    y1, y2 = max(0, y_min - pad), min(image_height - 1, y_max + pad)
    x1, x2 = max(0, x_min - pad), min(image_width - 1, x_max + pad)

    # calculate current and target aspect ratio
    box_h = y2 - y1 + 1
    box_w = x2 - x1 + 1
    target_ar = image_width / image_height
    box_ar = box_w / box_h

    # calculate extra pixels needed and space either side
    if box_ar != target_ar:
        if box_ar < target_ar:
            new_w = int(round(box_h * target_ar))
            delta = new_w - box_w
            space_bef, space_aft = x1, image_width - x2 - 1
        elif box_ar > target_ar:
            new_h = int(round(box_w / target_ar))
            delta = new_h - box_h
            space_bef, space_aft = y1, image_height - y2 - 1
        else:
            raise ValueError(f"Cannot handle aspect ratios: {box_ar}, {target_ar}")

        # calculate growth either side, targetting symmetry but guaranteeing aspect ratio
        if space_bef <= space_aft:
            grow_bef = min(delta // 2, space_bef)
            grow_aft = delta - grow_bef
        else:
            grow_aft = min(delta // 2, space_aft)
            grow_bef = delta - grow_aft

        # update bounding box locations
        if box_ar < target_ar:
            x1 -= grow_bef
            x2 += grow_aft
        else:
            y1 -= grow_bef
            y2 += grow_aft

    # check aspect ratio is within rounding range
    low_ar = (x2 - x1 + 0.5) / (y2 - y1 + 1.5)
    high_ar = (x2 - x1 + 1.5) / (y2 - y1 + 0.5)
    assert low_ar <= target_ar <= high_ar

    return [int(x1), int(y1), int(x2), int(y2)]
