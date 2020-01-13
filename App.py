import argparse
import os
import cv2
import sys
from LicensePlateRemoval import LicensePlateRemoval
import Config

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def check_basename(name: str, ext: str):
    return (not name.endswith('.back_up')) and (ext in ['.jpg', '.jpeg', '.png'])


def process_file(input_path, output_path):
    in_name, in_ext = os.path.splitext(os.path.basename(input_path))
    out_name, out_ext = os.path.splitext(os.path.basename(output_path))

    in_check = check_basename(in_name, in_ext)
    out_check = check_basename(out_name, out_ext)

    print('input path = {}'.format(input_path))
    print('output path = {}'.format(output_path))
    if in_check and out_check:
        im = cv2.imread(input_path)
        try:
            out_im, _ = license_plate_removal.image_remove(im)
            cv2.imwrite(output_path, out_im)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Detect and remove license plate")
    argparser.add_argument('-m', '--mode', default='local')
    argparser.add_argument('-i', '--input_path', required=True,
                           help='The local path to input images (file or directory).')
    argparser.add_argument('-o', '--output_path', default='output',
                           help='The local path where output images were contained (file or directory).')
    argparser.add_argument('-b', help='whether we need to create back up.', action='store_true')
    args = argparser.parse_args()

    print(args.mode)
    Config.Config(args.mode)
    license_plate_removal = LicensePlateRemoval()

    input_path = args.input_path
    output_path = args.output_path

    if output_path is None:
        output_path = input_path
    process_file(input_path, output_path)
