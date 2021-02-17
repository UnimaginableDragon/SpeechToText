from __future__ import absolute_import, division, print_function

import zipfile
import subprocess
import os
import shutil
from distutils.dir_util import copy_tree
#subprocess.call(['./package_install_script.sh'])
#with zipfile.ZipFile("Data/dev-clean.zip","r") as zip_ref:
#    zip_ref.extractall("Data")

import sys
paths = sys.argv[1:]

"""
paths = ['./Data/data.zip', './Data/Train/Best_hyperparameter_80_percent/', 
'./Data/Validation/Validation_10_percent/', './Data/Test/Test_10_percent/', 
'./Data/Train/Under_10_min_training/', './Data/Train/Under_90_min_tuning/', 
'./Data/Validation/3_samples/']
"""
with zipfile.ZipFile(paths[0],"r") as zip_ref:
    if not os.path.exists("Tmp"):
        os.mkdir("Tmp")
    zip_ref.extractall("Tmp")  # remove it at last
    
    
s = os.path.normpath(paths[0]).split(os.path.sep)
print(s)

data_folder_name = s[0]
data_zipfile_name = s[1]

listing = os.listdir("Tmp")  # should contain only one folder
actual_data_folder_name = listing[0]  # only one element, which is the first element

ten_min_train_folders = 0
ninety_min_train_folders = 0
train_folders = 0
validation_folders = 0
test_folders = 0


actual_data_folder_location = os.path.join("Tmp", actual_data_folder_name)

listing = os.listdir(actual_data_folder_location)
listing.sort(key=int)  # increasing integer ordering

for folder in listing:
    if ten_min_train_folders < 6:  # train under 10 mins
        copy_tree(os.path.join(actual_data_folder_location, folder), os.path.join(paths[4],"data",folder))
        ten_min_train_folders += 1
    
    if ninety_min_train_folders < 28:  # train under 90 mins
        copy_tree(os.path.join(actual_data_folder_location, folder), os.path.join(paths[5],"data",folder))
        ninety_min_train_folders += 1
        
    if train_folders < 33:  # train 
        copy_tree(os.path.join(actual_data_folder_location, folder), os.path.join(paths[1],"data",folder))
        train_folders += 1
     
    if train_folders == 33:
        if validation_folders == 0:
            # create 3 samples validation set
            # first copy all, then remove
            copy_tree(os.path.join(actual_data_folder_location, folder), os.path.join(paths[6],"data",folder))
            speaker_list = os.listdir(os.path.join(paths[6],"data",folder))
            for speaker in speaker_list[1:]:  # remove all speakers except the first one
                shutil.rmtree(os.path.join(paths[6],"data",folder,speaker))
                
            # now there is only one speaker. Keep only 3 wav files.
            speaker = speaker_list[0]
            file_list = os.listdir(os.path.join(paths[6],"data",folder, speaker))
            k=0
            transcription_file_name = None
            wav_files_name_list = []
            for file in file_list:
                if file.endswith(".wav"):
                    k+=1
                    if k>3:
                        os.remove(os.path.join(paths[6],"data",folder, speaker,file))
                    else:
                        wav_files_name_list.append(file)
                else:
                    transcription_file_name = file  # txt file containing transcription
                        
            valid_lines = []
            for line in open(os.path.join(paths[6],"data",folder, speaker,transcription_file_name), "r"):
               # for each line in the transcription file
                split = line.strip().split()
                file_id = split[0]
                if file_id+".wav" in wav_files_name_list:
                    valid_lines.append(line)
                   
            with open(os.path.join(paths[6],"data",folder, speaker,transcription_file_name), "w") as f:
                f.writelines(valid_lines)  # overwrite
            
        if validation_folders < 3:  # validation
            copy_tree(os.path.join(actual_data_folder_location, folder), os.path.join(paths[2],"data",folder))
            validation_folders += 1
                  
    if validation_folders == 3:
        if test_folders < 4:  # test
            copy_tree(os.path.join(actual_data_folder_location, folder), os.path.join(paths[3],"data",folder))
            test_folders += 1
            
shutil.rmtree("Tmp")
       

for i in range(1,7):
    folder_name = os.path.join(paths[i],"data")
    shutil.make_archive(folder_name, 'zip', folder_name)
    shutil.rmtree(folder_name)



# last 7 folders should go to validation and test set
# last 4 folders should go to test set, previous 3 folders should go to validation set
# first 6 folders -> 10 min train
# first 28 folders -> 90 min train
# first 33 folders -> train


"""
with open("in.txt") as f:
    lines = f.readlines()
    lines = lines[:3]  # 3 lines for 3 samples
    with open("out.txt", "w") as f1:
        f1.writelines(lines)


from shutil import copyfile

copyfile(src, dst)

import os
i=0
for file in os.listdir("/mydir"):
    if file.endswith(".wav"):
        copyfile(src, dst)
        i=i+1
    
    if i==3:
        break

"""



"""
for d in *; do find $d -type f -name "*.wav" | wc -l;done

73
94
42
59
64
75
58
77
52
55
95
58
87
75
57
38
47
90
49
64
41
101
57
36
59
59
83
80
78
74
96
55
77
71
82
72
78
75
65
55


"""


