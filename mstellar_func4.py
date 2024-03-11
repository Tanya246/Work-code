#the code intends to see upto which point in stellar mass, the compensation can be accounted for.
#first we divide the total galaxies in n stellar bins, derive their massfunction then start applying cut on it at MHI=7,7.5,8,8.5
#we have distribution of HI mass in each stellar bin which resembles a cumulative gaussian with different mean and sigma for each bin,fitting is done in another code and we are borrowing the parameters here
#we also have log mHI-log veff relation, which we are importing here.

#required packages
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.special import erf
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib import rcParams

# Enable LaTeX rendering
rcParams['text.usetex'] = True

#distribution of HI masses in each stellar bin
def integrated_gaussian(x, mean, stddev):
    return (0.5 * (1 + erf((x - mean) / (stddev * np.sqrt(2)))))

def inverse_error(x, mean, stddev):
    return mean + stddev * 2**0.5 * erfinv(2 * x - 1)

datafile="binned_data.txt" #mHI_veff relation
x_mhi=np.genfromtxt(datafile,usecols=1)#HI_masses
y_means=np.genfromtxt(datafile,usecols=0)#veff mean
y_errors=np.genfromtxt(datafile,usecols=2)#veff rms

num_arrays = 1000 #no of realisation
num_bin=15 #no of stellar bins
datafile="modified_max_dists_Taylor_f_cut_6.txt"#original data cat
x=np.genfromtxt(datafile,usecols=1)#stellar masses
y=np.genfromtxt(datafile,usecols=2)#HI masses
veff=np.genfromtxt(datafile,usecols=5)#veff for data

m_min= 4.4 #lowest stellar mass 
m_max= 12.6   #highest stellar mass
st_width = (m_max-m_min)/num_bin
st_width = round(st_width,5)
result_col=[]
var_col=[]
err_up=[]
err_down=[]
summ_col=[]
#for each bin we will use different seed to generate random number,the seed will be read from text file.
seed_filename="seed.txt"
seed_arr = np.genfromtxt(seed_filename,usecols=0,dtype=int)
#print(seed)
upper_cutoff = 6.76627 

param_filename = "parameter.txt"
mean_par = np.genfromtxt(param_filename,usecols=1)
sigma_par = np.genfromtxt(param_filename,usecols=2)
upper_cut=7
lower_cut=5.0
filename_cut = f"mfs_veff_compare1_{upper_cut}.dat"
num_gal_cut =np.genfromtxt(filename_cut,usecols=6) 
veff_cut =np.genfromtxt(filename_cut,usecols=4) 

data = np.loadtxt(filename_cut)
#print(data)
for i in range(num_bin):
    cols = int(num_gal_cut[i]) #number of remaining galaxies
    if(cols == 0):
        summ = veff_cut[i]
        var = 0
        log_sum = np.log10(summ +var)-np.log10(summ) 
        log_diff = np.log10(summ) - np.log10(summ-var) 
        result_col.append(np.where(log_sum > log_diff, log_sum, log_diff))
        var_col.append(var)
        summ_col.append(summ)
        err_up.append(np.log10(summ)+np.where(log_sum > log_diff, log_sum, log_diff))
        err_down.append(np.log10(summ)-np.where(log_sum > log_diff, log_sum, log_diff))
    else:
        seed = seed_arr[i]
        #print(seed)
        np.random.seed(seed)#to reproduce the result
        summ=0
        index_8 = integrated_gaussian(upper_cut,mean_par[i-1],sigma_par[i-1])
        index_5 = integrated_gaussian(lower_cut,mean_par[i-1],sigma_par[i-1])
        #print(index_8,index_5)
        rows = num_arrays
        #print(cols)
        veff_array = [None for _ in range(cols*num_arrays)]
        filename1 = f'bin{i+1}_n1_gaussian_total_{lower_cut}_{upper_cut}.txt'
        filename2 = f'veff_bin{i+1}_gaussian_total_{lower_cut}_{upper_cut}.txt'
        with open(filename1, 'w') as f:
            for j in range(cols*num_arrays):
                x = np.random.uniform(index_5,index_8,1)
                #print(x)
                #print(mean_par[i-1],sigma_par[i-1])
                #print(x,inverse_error(x,mean_par[i-1],sigma_par[i-1]))
                y_arrays = np.clip(np.random.normal(y_means, y_errors, size=len(y_means)),a_min=None,a_max=upper_cutoff)

                #print(y_arrays)
                cs1 = CubicSpline(x_mhi,y_arrays)
                veff_array[j]=1/(10**(cs1(inverse_error(x,mean_par[i-1],sigma_par[i-1]))))
                #print(10**(cs1(inverse_error(x,mean_par[i-1],sigma_par[i-1]))),inverse_error(x,mean_par[i-1],sigma_par[i-1]),x)
                f.write(str(10**(cs1(inverse_error(x,mean_par[i-1],sigma_par[i-1])))) + ' ' + str(inverse_error(x,mean_par[i-1],sigma_par[i-1])) + ' ' + str(x) + '\n')
        sums = []
        summ = 0
        veff_gaussian = []
        #combining generated veff in combination of number of galaxies mixed to get mfs realisation 
        for k in range(0,len(veff_array),cols):
            sum_column = sum(veff_array[k:k+cols])
            sums.append(sum_column / st_width)
        #storing mfs realisation in a seperate file 
        with open(filename2, 'w') as f:
            for sum_value in sums:
                #print(sum_value+veff_cut[i])
                modified_value = sum_value + veff_cut[i]
                veff_gaussian.append(modified_value)
                summ = summ + sum_value + veff_cut[i] 
                f.write(str(modified_value)+'\n')
        summ = summ/num_arrays
        #print(summ)
        var = 0
        #calculating rms wrt avarge for mfs realisations
        for k in range(len(veff_gaussian)):
            var = var + pow((veff_gaussian[k] - summ),2)
        var = np.sqrt(var/num_arrays)
        #print(var)

# Re    ad the input file into a NumPy array

# Pe    rform the calculation for each row
        log_sum = np.log10(summ +var)-np.log10(summ) 
        log_diff = np.log10(summ) - np.log10(summ-var) 
        result_col.append(np.where(log_sum > log_diff, log_sum, log_diff))
        var_col.append(var)
        summ_col.append(summ)
        err_up.append(np.log10(summ)+np.where(log_sum > log_diff, log_sum, log_diff))
        err_down.append(np.log10(summ)-np.where(log_sum > log_diff, log_sum, log_diff))
        #print(result)
#for i in range(num_bin):
#    print(summ_col[i],var_col[i],result_col[i],err_up[i],err_down[i])

# Stack the existing data array with new columns
data_with_new_column = np.column_stack((data, summ_col, var_col,result_col,err_up, err_down))

print(data_with_new_column)
# Specify the format for each column
formats = ['%.6f', '%g', '%d', '%.6f', '%g', '%d', '%d', '%g', '%.6f', '%.6f', '%.6f', '%.6f']

# Save the modified data back to the file
modified_data_file = f"mfs_veff_compare1_newf_{upper_cut}.dat"
np.savetxt(modified_data_file, data_with_new_column, fmt=formats)

