import os
import numpy as np
from datetime import date
from astropy.io import fits
from astropy.stats import sigma_clip


def get_all_data_with_exposure(
        exposure, path="./data/unprocessed/"):
    data_list, headers = [], []
    for filename in os.listdir(path):
        name_elements = filename.split('_')
        for elem in name_elements:
            if elem == str(round(exposure, 5)) + 's':
                fitsfile = fits.open(path + filename)
                data_list.append(sigma_clip(fitsfile[0].data))
                headers.append(fitsfile[0].header)
                break
    return np.stack(data_list, axis=0), headers


def build_master_header(sample_header, image_count):
    header = fits.PrimaryHDU().header
    header["DATE"] = date.isoformat(date.today())
    if "DATE-OBS" in sample_header:
        header["DATE-OBS"] = sample_header["DATE-OBS"].split('T')[0]
    for keyword in ["IMAGETYP", "ROWORDER", "EXPTIME", "ISOSPEED",
                    "INSTRUME", "TELESCOP", "FILTER", "OBJECT"]:
        if keyword in sample_header:
            header[keyword] = sample_header[keyword]
    header["SNAPSHOT"] = image_count
    return header


def srgb_to_grayscale(srgb):
    return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2]


def find_center_of_mass(data):
    y, x = np.ogrid[0 : data.shape[-2], 0 : data.shape[-1]]
    while len(data.shape) > len(x.shape):
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
    first_mement_x = x * data
    first_moment_y = y * data
    mean_mass = np.mean(data, axis=(-2, -1))
    return (np.mean(first_mement_x, axis=(-2, -1)) / mean_mass,
            np.mean(first_moment_y, axis=(-2, -1)) / mean_mass)
