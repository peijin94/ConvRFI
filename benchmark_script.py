import os

t_flag_range = 100000
chunk_len_lst = [1000,2000,5000,10000,20000]
device_lst = ['cpu','cuda']

import sys
this_dir = os.getcwd()
sys.path.append(this_dir)
work_dir = this_dir

os.chdir(work_dir) # go back home

for device_this in device_lst:
    for chunk_len_this in chunk_len_lst:
        os.chdir(work_dir) # go back home
        fname = 'p_'+device_this+'_c'+str(chunk_len_this)+'_t'+str(t_flag_range) 
        cmd_run = ("kernprof -l -o "+fname+" -v -u 1e-3 -z benchmark.py "+
            " --chunk_len "+ str(chunk_len_this) + " --t_flag_range "+ str(t_flag_range)+
            " --device "+device_this+
            " > "+fname+".txt")
        print(cmd_run)

        os.system(cmd_run)
        