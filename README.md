# DVGLicensePlateRemove
- Run API Server:

python3.5 API.py -m prod 

- Input API with parameter:

http://172.16.0.73:5994/ALPR?input=/opt/tuandh/licensePlateRemoveRunner/LicensePlateRemove/temp/2-63794.jpg&output=/opt/tuandh/licensePlateRemoveRunner/LicensePlateRemove/output/1-test.jpg

input_path = Đường dẫn file đến ảnh đầu vào

output_path = Đường dẫn của file ảnh muốn lưu

- Run App in commandline:

python3.5 API.py -m prod --input_path <đường dẫn đến file ảnh đầu vào> --output_path <đường dẫn của file ảnh muốn lưu>
