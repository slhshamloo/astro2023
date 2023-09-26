import numpy as np
import subprocess, os
from datetime import datetime
from netpbmfile import NetpbmFile
from astropy.io import fits
import piexif
from PIL.ExifTags import TAGS


RAW_DIR = "./raw/"
FITS_DIR = "./fits/"

OBSERVERS = "Shamloo_Yeganeh_Khanali" # "RezaRezaei"
FILTERNAME = "V"
OBSOBJ = "Dark"
IMAGETYP = "Dark Frame"
TELESCOPE = "Newton8"
LOCATION = "Aznaveh"

# -c: standard output
# -v: print verbose messages
# -w: use camera white balance
# -4: linear 16-bit output
# -q 4: RCD demosaicing
# -S 65535: set saturation level to maximum possible value for no data loss
DCRAW_OPTIONS = ("./dcraw", "-c", "-v", "-w", "-4", "-q", "4", "-S", "65535")


def get_metadata(filepath):
    metadata = piexif.load(filepath)
    datetime = metadata["0th"][piexif.ImageIFD.DateTime].decode()
    exposure = metadata["Exif"][piexif.ExifIFD.ExposureTime]
    return (
        ("DATE-OBS", convert_datetime_fits(datetime)),
        ("EXPTIME", round(exposure[0] / exposure[1], 5)),
        ("ISOSPEED", metadata["Exif"][piexif.ExifIFD.ISOSpeedRatings]),
        ("INSTRUME", metadata["0th"][piexif.ImageIFD.Model].decode())
    )


def get_fits_name(metadata, has_filter=True, add_location_name=True,
                  add_telescope_name=True):
    filterstr = '_' + FILTERNAME[0] if has_filter else ''
    locationstr = '_' + LOCATION if add_location_name else ''
    telescopestr = '_' + TELESCOPE if add_telescope_name else ''
    return (OBSERVERS + filterstr + '_' + OBSOBJ + '_' + str(metadata[1][1])
            + 's' + '_ISO' + str(metadata[2][1])
            + '_' + metadata[3][1].split(' ')[-1] + telescopestr
            + '_' + convert_datetime_filename(metadata[0][1])
            + locationstr + ".fits")


def get_raw_data(filepath):
    with open("temp.pam", "wb") as netpbm_output:
        subprocess.run(DCRAW_OPTIONS + (filepath,), stdout=netpbm_output)
    with NetpbmFile("temp.pam") as netpbm_file:
        # Read data and reshape to correct FITS dimensions
        data = netpbm_file.asarray().transpose().swapaxes(1, 2)
    if data.shape[-2] > data.shape[-1]:
        data = np.rot90(data, 3, (-2, -1))
    os.remove("temp.pam")
    return data


def set_fits_header(header, metadata, has_filter=True,
                    add_telescope_name=True, add_object_name=True):
    while len(header) < (2 * 36 - 1):
        header.append()
    header["IMAGETYP"] = IMAGETYP
    header["DATE"] = datetime.utcnow().isoformat()[:-7]
    for (keyword, value) in metadata:
        header[keyword] = value
    if add_telescope_name:
        header["TELESCOP"] = TELESCOPE
    if has_filter:
        header["FILTER"] = FILTERNAME
    if add_object_name:
        header["OBJECT"] = OBSOBJ
    header["OBSERVER"] = ", ".join(OBSERVERS.upper().split('_'))


def convert_datetime_fits(metadata_datetime):
    date, time = metadata_datetime.split(' ')
    date = '-'.join(date.split(':'))
    return date + 'T' + time


def convert_datetime_filename(fits_datetime):
    date, time = fits_datetime.split('T')
    date = ''.join(date.split('-'))
    time = ''.join(time.split(':'))
    return date + 'T' + time


def main():
    for filename in os.listdir("./raw/"):
        filepath = RAW_DIR + filename
        metadata = get_metadata(filepath)
        fitsname = get_fits_name(metadata, False, False, False)
        fitsfile = fits.PrimaryHDU(data=get_raw_data(filepath))
        set_fits_header(fitsfile.header, metadata, False, False, False)
        fitsfile.writeto(FITS_DIR + fitsname, overwrite=True)


if __name__ == "__main__":
    main()
