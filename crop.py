import operator

boundaries = {  # fronts are more frequent here
    'min_x': 60,
    'max_x': 322,
    'min_y': 19,
    'max_y': 275,
}


def crop_center(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def crop_boundaries(imgs):
    return imgs[:, boundaries['min_y']:boundaries['max_y'] + 1, boundaries['min_x']:boundaries['max_x'] + 1, :]


def crop_2d(inp):
    return inp[boundaries['min_y']:boundaries['max_y'] + 1, boundaries['min_x']:boundaries['max_x'] + 1]
