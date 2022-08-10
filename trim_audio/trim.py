#!/usr/bin/env python
import os
import sys
import glob
import subprocess
import time

# get Current Working Directory (CWD)
# pwd = os.getcwd()
# pwd = pwd+""
# print(pwd)
# # get list of files 
# if len(sys.argv) > 1:
#     filenames = sys.argv[1:] # Linux
#     #filenames = glob.glob(sys.argv[1]) # Windows

# else:
#     filenames = glob.glob("*.mp3")

filenames = glob.glob("*.mp3")
print(filenames)
x = 0
for filename in filenames:
    x += 1
    print(x)
    video = str(x)
    name = filename.split(".")
    video_name =name[0]

    # print(pwd+"/"+filename)
    print(filename)
    cmd = "ffmpeg -ss 0 -t 12 -i "+filename+" trim_alaq/"+ filename
    print(str(cmd))
    cd = str(cmd)
    os.system(cd) 

#     cmd = "ffmpeg -ss 00:00:00 -t 00:12:00 -i "+filename+" -c copy "+pwd+"/trim_alaq/"+ filename
#     print(cmd)

#     # run ffmpeg 

#     cmd = "ffmpeg -ss 00:00:00 -t 00:12:00 -i"+ filename+" -c copy "+pwd+"/trim_alaq/"+ filename"
#     cmd ="ffmpeg -ss 00:00:00 -t 00:12:00 -i audio.mp3 -c copy audio-trimmed.mp3"


# cmd = "ffmpeg -ss 0 -t 12 -i 096-al-alaq.mp3 trim_alaq/1.mp3"
# print(str(cmd))
# cd = str(cmd)
# os.system(cd) 
