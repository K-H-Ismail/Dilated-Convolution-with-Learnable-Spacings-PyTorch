import torch
import time
from math import sqrt
import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DCLS.functions.spmm_functionnal import sparse_mm_dense


'''print('Cusparse mm : Forward: {:.3f} s'.format(forward * 1e1/1e1))

if torch.all(res_torch == res_cusparse) : 
    print("results of native mm and cusparse mm are equal.") 
else : 
    print("results of native mm and cusparse mm are not equal ! norm1 error : {:.3f}".format((res_native - res_cusparse).norm()/(n*n)) )'''
           
           
def measure_method(sparse, dense, method):
    forward = 0       
    start = time.time()
    if method == 'native':
        res_torch=torch.mm(sparse,dense)
    elif method == 'torch-sparse':
        res_torch=torch.mm(sparse.to_sparse(),dense)
    elif method == 'cusparse':
        res_cusparse=sparse_mm_dense(sparse,dense)          
    torch.cuda.synchronize()
    forward = time.time() - start           
    return forward

def measures(sparse, dense):
    return (1000*measure_method(sparse, dense, 'native'),
            1000*measure_method(sparse, dense, 'cusparse'))
           
def create_matrice(m,n,density):
    range_density_m = int(m*sqrt(density))
    range_density_n = int(n*sqrt(density))    
    sparse = torch.zeros(m,n).cuda()
    sparse[:range_density_m,:range_density_n] = torch.rand(range_density_m,range_density_n).cuda()
    r = torch.randperm(m).cuda()
    c = torch.randperm(n).cuda()
    sparse = sparse[r][:,c] 

    return sparse

def timings(n_max, d_max,step):
    print("n,density (%),tnative (ms),tcusp (ms)")    
    for n in range(int(n_max*step),n_max+int(n_max*step),int(n_max*step)):
        dense = torch.rand(n,n).cuda()
        sparse = create_matrice(n,n,0.001)
        t_dense = 1000*measure_method(sparse, dense, 'native')          
        for density in torch.linspace(0.001,d_max,int(1/step)):
            sparse = create_matrice(n,n,density)
            print("{:d},{:.1f},{:.3f},{:.3f}".format(n,100*density,t_dense,1000*measure_method(sparse, dense, 'cusparse')))
          

           
def test():
    timings(20000,0.1,1/100)
    
#test()
'''sparse, dense = create_matrice(2048,256*3*3*16*16,4/(16*16)), torch.rand(256*3*3*16*16,16*33*33).cuda()
print(measures(sparse, dense))
sparse, dense = create_matrice(2048,256*3*3*16*16,4/(16*16)), torch.rand(256*3*3*16*16,16*33*33).cuda()
print(measures(sparse, dense))
sparse, dense = create_matrice(2048,256*3*3*16*16,4/(16*16)), torch.rand(256*3*3*16*16,16*33*33).cuda()
print(measures(sparse, dense))
sparse, dense = create_matrice(2048,256*3*3*12*12,4/(12*12)), torch.rand(256*3*3*12*12,16*33*33).cuda()
print(measures(sparse, dense))
sparse, dense = create_matrice(2048,256*3*3*12*12,4/(12*12)), torch.rand(256*3*3*12*12,16*33*33).cuda()
print(measures(sparse, dense))
sparse, dense = create_matrice(2048,256*3*3*12*12,4/(12*12)), torch.rand(256*3*3*12*12,16*33*33).cuda()
print(measures(sparse, dense))'''

'''
!pip install plotly
import pandas as pd
import numpy as np 
import plotly.graph_objects as go

df = pd.read_csv("measurement_top.csv")



df = df[df["speedup"] > 2]
print(df)


def df_to_plotly(df):
    return {'z': np.log(df["speedup"]).tolist(),
            'x': df["density (%)"].tolist(),
            'y': df["n"].tolist()}



fig = go.Figure(data=go.Heatmap(df_to_plotly(df)))
fig.show()'''

'''import pandas as pd
df = pd.read_csv("measurement_20k.csv")
df['speedup'] = df['tnative (ms)'] / df['tcusp (ms)']
print(df)
df.to_csv("measurement_20k.csv",index=False)'''