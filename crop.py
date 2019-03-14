import operator

# boundaries = {  # on 100 km wide fronts
#     'min_x': 0,
#     'max_x': 340,
#     'min_y': 18,
#     'max_y': 276,
# }

# boundaries = {  # on 300 km wide fronts
#     'min_x': 0,
#     'max_x': 344,
#     'min_y': 15,
#     'max_y': 276,
# }

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