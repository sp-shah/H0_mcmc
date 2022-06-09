import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib
from matplotlib import pyplot as plt
import sys
import matplotlib.cm as cm
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
import emcee
import scipy.optimize as op
import random
import numpy as np
import corner
import hconst_dict as hd
import h5py
from mpi4py import MPI
import corner

############################### MPI RUNS ####################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
total_ranks = 32
#################################################################
ndim, nwalkers = 1, 700
#data_pts =  [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,175,200]
data_pts = [80,120,140]#[1,2,4,10,20,30,40,50,60,80,100,120]
runs = 3 #each processor will run once - number of processors will determine how many runs
burn = 70
d_err = 0.05
nsteps = 800
type_num = np.int(sys.argv[1])
if rank == 0:
    path_to_simdata = "/home/shivani.shah/Projects/LIGO/analysis/mcmc/sim_data/"
    file_name = path_to_simdata+"hconst_w_mass_dist_60.hdf5"
    print("Will create data now")
    d,v,xar,yar,zar,vpec,logmass = hd.read_data_hdf5(file_name) #v=l.o.s total velocity; vpec=l.o.s peculiar velocity 
    print("Will call build tree now")
    tree = hd.build_tree(xar,yar,zar)
    print("Tree received")
    print("Will obtain stellar mass now")
    log_stellar_mass = hd.shm(logmass)    
    data = dict(d = d, v = v, xar = xar, yar = yar, zar = zar, vpec = vpec, log_stellar_mass = log_stellar_mass, tree = tree)
else:
    data = None

data = comm.bcast(data, root = 0)
d = data["d"]; v = data["v"]; xar = data["xar"]; yar = data["yar"]; zar = data["zar"]; vpec = data["vpec"]; log_stellar_mass = data["log_stellar_mass"]; tree = data["tree"]

###################################################################################################
for data_pt in data_pts:
    print("Detection no "+np.str(data_pt))
    samples = []
    flatlnprob = []
    dist_ar = []
    quant50_ar= []
    quant16_ar = []
    quant84_ar = []
    dist0_ar = []
    vel0_ar = []
    vel_ar = []
    dist_err_ar = []
    vel_err_ar = []
    num = 0
    while num < runs:
        if rank == 0: print("Run number "+np.str(num))
        num +=1 
        dist0, vel0,x,y,z = hd.filter_data_w_stellarmass(d,v,xar,yar,zar,log_stellar_mass,data_pt)
        #dist0, vel0, x, y, z = hd.filter_data(d,v,xar,yar,zar,data_pt)
        pec, pec_sigma = hd.obtainpecvel(tree, x, y, z,vpec)
        dist, vel, dist_err, vel_err = hd.final_data(dist0, vel0, d_err, pec, pec_sigma, type_num)
        pos =  [90. + 1.e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, hd.lnprob, args = (dist, vel, dist_err, vel_err))
        sampler.run_mcmc(pos,nsteps)
    
        flatchain = sampler.chain[:,30:,:].reshape(1,-1)[0] #1-D list/array
        flatlnprobability = sampler.lnprobability[:,30:].reshape(1,-1)[0] #1-D list/array

        samples.append(flatchain)
        flatlnprob.append(flatlnprobability)
        #Collecting all the data points
        dist_ar.append(dist)
        vel_ar.append(vel)
        dist0_ar.append(dist0)
        vel0_ar.append(vel0)
        dist_err_ar.append(dist_err)
        vel_err_ar.append(vel_err)
        
    if total_ranks > 1:
        samples_allrank = comm.gather(samples, root = 0)
        flatlnprob_allrank = comm.gather(flatlnprob, root = 0)
        dist_allrank = comm.gather(dist_ar, root = 0)
        vel_allrank = comm.gather(vel_ar, root = 0)
        dist0_allrank = comm.gather(dist0_ar, root = 0)
        vel0_allrank = comm.gather(vel0_ar, root = 0)
        dist_err_allrank = comm.gather(dist_err_ar, root = 0)
        vel_err_allrank = comm.gather(vel_err_ar, root = 0)
    

    if rank == 0:
        path = "/home/shivani.shah/Projects/LIGO/analysis/mcmc/runs/60_shm/0p15/output/case"
        f = h5py.File(path+np.str(type_num)+"_30runs.hdf5", "a")
        sub = f.create_group("dtpt" + np.str(data_pt))
        dist = sub.create_dataset("dist", dtype = float, data = np.array(dist_allrank))
        vel = sub.create_dataset("vel", dtype = float, data = np.array(vel_allrank))
        dist0 = sub.create_dataset("dist0", dtype = float, data = np.array(dist0_allrank))
        vel0 = sub.create_dataset("vel0", dtype = float, data = np.array(vel0_allrank))
        dist_err = sub.create_dataset("dist_err", dtype = float, data = np.array(dist_err_allrank))
        vel_err = sub.create_dataset("vel_err", dtype = float, data = np.array(vel_err_allrank))
        samples = sub.create_dataset("samples", dtype = float, data = np.array(samples_allrank))
        flatlnprob = sub.create_dataset("flatlnprob", dtype = float, data = np.array(flatlnprob_allrank))

########################################################################################################
#Incase we want to use mode or mean. Or we want to write to a text file
'''
max_ind = np.argmax(flatlnprob)
mode = samples[max_ind]
mode_lo, mode_up = hd.hpd(samples, 0.68)

mean = samples.mean(axis = 0)
std = samples.std(axis = 0)


datafile_path = "/home/shivani.shah/Projects/LIGO/analysis/results/dtptstest_"+np.str(type_num)+".txt"
        with open(datafile_path, "a") as datafile_id:
            datafile_id.write('\n%3.3f %3.3f %3.3f %3.3f %3.3f %3.3f %3.3f %3.3f' %(mode, mode_lo, mode_up, mean, std, quant50, quant16, quant84))
'''    
'''
    if total_ranks > 1:
        for i in range(total_ranks-1):
            comm.send(d, dest=i+1)
            comm.send(v, dest=i+1)
            comm.send(xar, dest=i+1)
            comm.send(yar, dest=i+1)
            comm.send(zar, dest=i+1)
            comm.send(vpec, dest=i+1)
            comm.send(log_stellar_mass, dest=i+1)
            comm.send(tree, dest=i+1)
        
if rank != 0:
    d = comm.recv(source = 0)
    v = comm.recv(source = 0)
    xar = comm.recv(source = 0)
    yar = comm.recv(source = 0)
    zar = comm.recv(source = 0)
    vpec = comm.recv(source = 0)
    log_stellar_mass = comm.recv(source = 0)
    tree = comm.recv(source = 0)
    
'''
'''
flatchain_corner = sampler.chain[:,30:,:].reshape(-1,1) #nested 1D array for cornerplot
fig = corner.corner(flatchain_corner, quantiles = (0.16,0.50,0.84))
        fig.savefig("results/corner_shm_60.png")
'''
'''
if total_ranks > 1:
            samples = np.array(samples_allrank).reshape(1,-1)[0]
            flatlnprob = np.array(flatlnprob_allrank).reshape(1,-1)[0]
        else:
            samples = np.array(samples).reshape(1,-1)[0]
            flatlnprob = np.array(flatlnprob).reshape(1,-1)[0]

        quant16_av, quant50_av,quant84_av = np.percentile(samples, [16.,50.,84.])
'''
'''
 quant50_av = sub.create_dataset("quant50_av", dtype = float, data = quant50_av)
        quant16_av = sub.create_dataset("quant16_av", dtype = float, data = quant16_av)
        quant84_av = sub.create_dataset("quant84_av", dtype = float, data = quant84_av)
        quant50 = sub.create_dataset("quant50", dtype = float, data = np.array(quant50_allrank).reshape(1,-1)[0])
        quant16 = sub.create_dataset("quant16", dtype = float, data = np.array(quant16_allrank).reshape(1,-1)[0])
        quant84 = sub.create_dataset("quant84", dtype = float, data = np.array(quant84_allrank).reshape(1,-1)[0])
'''
