import string
import torch
import os,re
from tqdm import tqdm
from RFIconvFlag import *
import platform
import h5py
if platform.system()=='Windows':
    h5dir = 'D://H5/'
else:
    h5dir = './H5/'

h5_fname = 'L860566_SAP000_B000_S0_P000_bf.h5'
import sys
this_dir = os.getcwd()
sys.path.append(this_dir)
work_dir = this_dir

os.chdir(work_dir) # go back home
os.chdir(h5dir)
m = re.search('B[0-9]{3}', h5_fname)
m.group(0)
beam_this = m.group(0)[1:4]
m = re.search('SAP[0-9]{3}', h5_fname)
m.group(0)
SAP = m.group(0)[3:6]
f = h5py.File( h5_fname, 'r' )

device ='cpu'
device = "cuda:0" if torch.cuda.is_available() else "cpu"




from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--freq_window', help=' ', type=int, default=16)
    parser.add_argument('--time_window', help=' ', type=int, default=32)
    parser.add_argument('--chunk_len', help=' ', type=int, default=96*60)
    parser.add_argument('--t_flag_range', help=' ', type=int, default=100000)
    parser.add_argument('--device', help=' ', type=str, default=device)

    return parser.parse_args()

@profile
def benchmark_func(freq_window=16,time_window=32,chunk_len = 96*60,t_flag_range = 100000,device=device):
    
    net = RFIconv(device=device)
    net = init_RFIconv(net,aggressive_factor = [1.6,1.8,0.5,0.5],device=device)

    idx_all_run = int((t_flag_range)/chunk_len)    
    big_arr = []

    if device!='cpu':
        torch.cuda.empty_cache() 
    os.chdir(h5dir)
    
    for num_chunk in tqdm(range(idx_all_run)):
        data_test=f['SUB_ARRAY_POINTING_'+SAP+'/BEAM_'+beam_this+'/STOKES_0'][
            chunk_len*num_chunk:chunk_len*(num_chunk+1),:]
        
        with torch.no_grad():# no grad:
            data_tmp_cpu =  torch.tensor(data_test[None,None,:,:])
            data_tmp = data_tmp_cpu.to(device)
            output = net(data_tmp)
            output_flag_cpu = output.cpu()
            output_flag_float = 1- output_flag_cpu.to(dtype=torch.float32).squeeze()[None,None,:,:]

            conv_down_after_flag =  F.conv2d(data_tmp_cpu*(output_flag_float),
                    torch.ones([1,1,time_window,freq_window])/freq_window/time_window,
                    stride=(time_window,freq_window),padding=(0,0)).squeeze()

            conv_down_weight_after_flag =  F.conv2d(output_flag_float,
                    torch.ones([1,1,time_window,freq_window])/freq_window/time_window,
                    stride=(time_window,freq_window),padding=(0,0)).squeeze()

            small_arr  =  conv_down_after_flag/conv_down_weight_after_flag
            if device!='cpu':
                torch.cuda.empty_cache() 
        big_arr.append(small_arr)
    os.chdir(work_dir)
    return big_arr,output_flag_float,output_flag_cpu


def main():
    args = parse_args()
    benchmark_func(
            args.freq_window,
            args.time_window,
            args.chunk_len,
            args.t_flag_range,
            args.device)

if __name__ == '__main__':
    main()
