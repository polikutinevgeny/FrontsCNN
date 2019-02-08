mean_std = {  # calculated on available dates
    'air': (284.80401611328125, 14.781936645507812),
    'mslet': (101505.328125, 927.8274536132812),
    'shum': (0.009227436035871506, 0.005920239724218845),
    'uwnd': (-0.2842417359352112, 5.01072883605957),
    'vwnd': (-0.3550795614719391, 4.506356716156006)
}


def normalize(a, name):
    mean, std = mean_std[name]
    return (a - mean) / std