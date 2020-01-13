from flask import Flask, request
from flask_restful import Resource, Api
from flask.json import jsonify
import os
import cv2
from LicensePlateRemoval import LicensePlateRemoval
import PIL as pil
import numpy as np
import os
from flask import Flask, flash, request, redirect, url_for, render_template, make_response, Response, send_file
import Config
from datetime import datetime
import pytz
import sys
import argparse
import logging

tz_HCM = pytz.timezone('Asia/Ho_Chi_Minh')

UPLOAD_FOLDER = 'temp'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'super secret key'
api = Api(app)


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


class ALPR(Resource):

    def get(self):
        args = request.args

        input_path = args['input']
        output_path = args.get('output', default=input_path, type=str)

        process_file(input_path, output_path)

        return 'Done request: \n - input = {}.\n output = {}'.format(input_path, output_path)


class ALPRUI(Resource):
    def allowed_file(self, filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def post(self):
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and self.allowed_file(file.filename):
            filename = file.filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            file.save(input_path)
            process_file(input_path, output_path)
            return send_file(output_path, mimetype='image/png')

        return Response('''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
              <input type=file name=file>
              <input type=submit value=Upload>
            </form>
            ''', mimetype='text/html')

    def get(self):
        return Response('''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype=multipart/form-data>
              <input type=file name=file>
              <input type=submit value=Upload>
            </form>
            ''', mimetype='text/html')


api.add_resource(ALPRUI, '/ALPRUI')

api.add_resource(ALPR, '/ALPR')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Detect and remove license plate")
    argparser.add_argument(
        '-m',
        '--mode',
        default='local')
    args = argparser.parse_args()
    Config.Config(args.mode)

    license_plate_removal = LicensePlateRemoval()
    app.run(host='0.0.0.0', port=5994)
