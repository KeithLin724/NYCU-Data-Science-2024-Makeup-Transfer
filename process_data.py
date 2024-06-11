import os
import random
import shutil

random.seed(0)
"""
list_ = [str(e) for e in range(121)]
out = open('test.txt', 'w')
out.write(' '.join(list_))
"""
input_path = "C:\\Users\\redman\\Downloads\\BeautyGAN-PyTorch-reimplementation-master\\mtdataset\\images"
out_path = "mtdataset/images"
if not os.path.exists(out_path):
    os.makedirs(out_path)
file1 = open("makeup_test.txt", "r")
non_makueps = file1.readlines()
for line in non_makueps:
    filename = line.split()[0]
    folder = filename.split("/")[0]
    subfolder = os.path.join(out_path, folder)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    print(filename)
    shutil.copyfile(
        os.path.join(input_path, filename), os.path.join(out_path, filename)
    )
