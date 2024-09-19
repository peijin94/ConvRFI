# ConvRFI
<img width="500" alt="image" src="https://github.com/user-attachments/assets/28e5de6e-0dc9-4537-81e1-894179eb47f3">


ConvRFI is an algorithm designed for radio frequency interference (RFI) flagging in solar and space weather low-frequency radio observations. It uses morphological convolution (shown as in Figure below) to detect RFI patterns in dynamic spectra. ConvRFI is particularly effective at preserving solar radio burst signals while flagging RFI. It employs convolution kernels to identify line-like and edge-like RFI features. The method is computationally efficient, GPU-enabled, and can process data at speeds up to 0.35 GB/s on a laptop GPU. 

<img width="380" alt="image" src="https://github.com/user-attachments/assets/468e1338-0bb6-4993-abc3-889552c5b1c6">


## Example

```python
from RFIconvFlag import  RFIconv,init_RFIconv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device!='cpu':
    torch.cuda.empty_cache()

net = RFIconv(device=device)
factor = [1.65,1.65,0.45,0.45]

net = init_RFIconv(net,aggressive_factor = agg_factor,device=device).to(device)
with torch.no_grad():
  output = net(torch.tensor(data_test.squeeze()[None,None,:,:]).to(device)).squeeze().cpu().numpy()
data_test_ma = np.ma.masked_where((output),data_test.squeeze())

if device!='cpu':
  torch.cuda.empty_cache() 
```


## cite as

Zhang, P., Offringa, A.R., Zucca, P., Kozarev, K. and Mancini, M., 2023. RFI flagging in solar and space weather low frequency radio observations. Monthly Notices of the Royal Astronomical Society, 521(1), pp.630-637.

```
@article{zhang2023rfi,
  title={RFI flagging in solar and space weather low frequency radio observations},
  author={Zhang, Peijin and Offringa, Andr{\'e} R and Zucca, Pietro and Kozarev, Kamen and Mancini, Mattia},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={521},
  number={1},
  pages={630--637},
  year={2023},
  publisher={Oxford University Press}
}
```

https://doi.org/10.1093/mnras/stad491
