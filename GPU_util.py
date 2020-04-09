# memory footprint support libraries/code
#!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi

import pip
import subprocess
import sys
import tensorflow as tf

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

        

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]        
        
if 'GPUtil' not in installed_packages:
    install('gputil')

if 'psutil' not in installed_packages:
    install('psutil')
    
if 'humanize' not in installed_packages:  
    install('humanize')

    
import psutil
import humanize
import os
import GPUtil as GPU

		
def GPU_info():    
   
    if( tf.test.is_gpu_available() ):
        GPUs = GPU.getGPUs()

        # XXX: only one GPU on Colab and isn’t guaranteed
        gpu = GPUs[0]

    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))

    if( tf.test.is_gpu_available() ):
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))