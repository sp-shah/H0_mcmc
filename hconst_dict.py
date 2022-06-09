import numpy as np
import matplotlib
#matplotlib.use('agg')
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
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import h5py


###############################################################################################

def read_data_with_pos(filename, **kwargs):
    if ("data" in kwargs) and ("data_pt" in kwargs):
        no_datapts = kwargs["data_pt"]
        data = kwargs["data"]
        d = data["dist"]
        v = data["vel"]
        xar = data["x"]
        yar = data["y"]
        zar = data["z"]
        ind = random.sample((np.arange(0,np.size(d),1)), no_datapts)
        dist = d[ind]; vel = v[ind]; x = xar[ind]; y = yar[ind]; z = zar[ind] 
    
        return np.array(dist), np.array(vel), np.array(x), np.array(y), np.array(z)
    else:
        print("About to build data")
        data = np.genfromtxt(filename, names = ["dist","vel","x","y","z","vpec"])
        print("data built with datapts = ", np.size(data))
        return data

###################################################################################################3
def read_data_hdf5(filename):
    data = h5py.File(filename,"r")
    group = data["Subhalo"]
    d = group["dist"][:]
    v = group["vel"][:]
    xar = group["x"][:]
    yar = group["y"][:]
    zar= group["z"][:]
    v_pec = group["vpec"][:]
    logmass = group["logmass"][:]
    data.close()

    return d, v, xar, yar, zar, v_pec, logmass
###################################################################################################3
def read_output_hdf5(filename, group_name):
    data = h5py.File(filename)
    group = data[group_name]
    dictionary = dict(dist = group["dist"][:], vel = group["vel"][:], dist0 = group["dist0"][:], vel0 = group["vel0"][:], dist_err = group["dist_err"][:], vel_err = group["vel_err"][:], samples = group["samples"][:], flatlnprob = group["flatlnprob"][:])
    
    data.close()
    return dictionary

##################################################################################################
def build_tree(xar,yar,zar):
    data = np.stack([xar, yar, zar], axis =1)
    tree = cKDTree(data)
    print("Tree Built")
    return tree

###################################################################################################33
def filter_data(d,v,xar,yar,zar,no_datapts):
    ind = random.sample((np.arange(0,np.size(d),1)), no_datapts)
    dist = d[ind]; vel = v[ind]; x = xar[ind]; y = yar[ind]; z = zar[ind]
    return dist,vel,x,y,z
#################################################################################################3
def shm(logmass):
    #source: https://iopscience.iop.org/article/10.1088/0004-637X/710/2/903/pdf
    #equation 2, table 1
    halo_mass = 10**(logmass)
    normalization = 0.02828
    beta = 1.057
    M_char = 10**(11.884)
    gamma = 0.556
    stellar_mass = 2*halo_mass*normalization*((halo_mass/M_char)**-beta + (halo_mass/M_char)**gamma)**(-1)
    log_stellar_mass = np.log10(stellar_mass)
    return log_stellar_mass
#################################################################################################
def stellar_mass_cdf(log_stellar_mass):
    x = np.arange(0, len(log_stellar_mass), 1)
    x = x.astype(np.int)
    y = np.cumsum(log_stellar_mass); y = np.insert(y,0,0.); y = np.delete(y,-1)
    norm = np.sum(log_stellar_mass); y /= norm; y = y.astype(np.float)
    f = interp1d(y, x)
    #values, bins = np.histogram(log_stellar_mass, bins=40)
    #cumulative = np.cumsum(values)
    #cumulative = cumulative.astype(np.float)
    #cumulative /= np.max(cumulative)
    range_cumulative = [np.min(y),np.max(y)]
    #print(range_cumulative)
    #f = interp1d(cumulative, bins[:-1])
    return f, range_cumulative
##################################################################################################
def filter_data_w_stellarmass(d,v,xar,yar,zar, log_stellar_mass, no_datapts):
    f, range_cumulative = stellar_mass_cdf(log_stellar_mass)
    y = np.random.uniform(range_cumulative[0], range_cumulative[1], no_datapts)
    #print("-------------------------------------------")
    #print("Random pick")
    #print(y)
    ind_mass = f(y).astype(np.int)
    #print("-------------------------------------------")
    #print("Mass Selected")
    mass_selected = log_stellar_mass[ind_mass]
    dist = d[ind_mass]
    vel = v[ind_mass]
    x = xar[ind_mass]
    y = yar[ind_mass]
    z = zar[ind_mass]
    #print(mass_selected)
    '''
    dist = []; vel = []; x = []; y = []; z = []
    for ele in mass_selected:
        idx = (np.abs(log_stellar_mass - ele)).argmin()
        dist.append(d[idx])
        vel.append(v[idx])
        x.append(xar[idx])
        y.append(yar[idx])
        z.append(zar[idx])
    '''   
    return np.array(dist), np.array(vel), np.array(x), np.array(y), np.array(z)

#####################################################################################################
def obtainpecvel(tree, x, y, z,vpec):
    data_com = np.stack([x,y,z],axis = 1)
    d, ii= tree.query(data_com, k = 100, distance_upper_bound = 30.)
    d = np.array(d); ii = np.array(ii)
    pec_ar = []; pec_sigma_ar = []
    for row in range(len(d[:,1])):
        drow = d[row,:]; iirow = ii[row,:]
        w = np.where(drow == np.float('inf'))
        dnew = np.delete(drow, w); iinew = np.delete(iirow, w)
        if np.size(dnew) == 0:
            pec_ar.append(np.nan); pec_sigma_ar.append(np.nan)
        else:
            pec_thispos = np.mean(vpec[iinew])
            pec_sigma_thispos = np.sqrt(np.var(vpec[iinew]))
            pec_ar.append(pec_thispos); pec_sigma_ar.append(pec_sigma_thispos)
            
    return pec_ar, pec_sigma_ar

#############################################################m#################################33
def final_data(dist0, vel0, d_err, pec, pec_sigma, type_num):
    if type_num==4:
        
        dist_err = dist0*d_err
        dscatter = np.array(np.random.normal(loc=0.0, scale=dist_err))
        dist = dist0 + dscatter
        ind = np.where(np.isnan(pec))[0]
        if np.size(ind) > 0:
            pec[ind] = 0.; pec_sigma[ind] = vel0[ind]*0.2
        vel = vel0 - pec
        #print("Final Vel")
        #print(vel)
        #print("--------------------------------------")
        vel_err = pec_sigma
        #vel = vel0
        return np.array(dist), np.array(vel), np.array(dist_err), np.array(vel_err)

    if type_num == 3:
        #print("Initial Distance" + np.str(dist0))
        #print("H0 "+np.str(np.array(vel0)/np.array(dist0)))
        dist_err = dist0*d_err
        #print("Dist Error" + np.str(dist_err))
        dscatter = np.array(np.random.normal(loc=0.0, scale=dist_err))
        #print("Dist Scatter"+ np.str(dscatter))
        dist = dist0 +dscatter
        #print("Final Distance" +np.str(dist))
        ind = np.where(np.isnan(pec_sigma))[0]
        if np.size(ind) > 0:
            pec_sigma[ind] = vel0[ind]*0.2
        vel_err = pec_sigma
        vel = vel0
        #print("Velocity "+np.str(vel))
        #print("H0 "+np.str(np.array(vel)/np.array(dist)))
        return np.array(dist), np.array(vel), np.array(dist_err), np.array(vel_err)

    if type_num == 2:
        dist_err = dist0*d_err
        dscatter = np.array(np.random.normal(loc=0.0, scale=dist_err))
        dist = dist0 + dscatter
        vel_err = np.full(np.size(dist),1.e-8)
        vel = vel0
        return np.array(dist), np.array(vel), np.array(dist_err), vel_err

    if type_num == 1:
        dist = dist0
        dist_err = np.full(np.size(dist),1.e-8)
        vel_err = np.full(np.size(dist),1.e-8)
        vel = vel0
        return np.array(dist), np.array(vel), dist_err, vel_err

        

############################################################################################3333333

def lnLike_hogg(Theta, d, v, dist_err, vel_err):

    m = np.copy(Theta)
    b = 0.0
    theta = np.arctan(m)
    v_unit = [-np.sin(theta), np.cos(theta)]
    v_unit_tran = np.transpose(v_unit)
    sigma_distance = dist_err
    sigma_redshift = vel_err
    sigma_distance_sq = sigma_distance**2
    sigma_redshift_sq = sigma_redshift**2
    cov = 0
    Z = np.array([d,v])
    del_array = np.matmul(v_unit_tran,Z) - b*np.cos(theta)
    a = np.arange(len(sigma_distance))
    sigma_red_cov = np.insert(sigma_redshift_sq, a, 0.)
    a += 1
    sigma_dist_cov = np.insert(sigma_distance_sq, a, 0.)
    S = np.array([sigma_dist_cov, sigma_red_cov])
    sigmasq_stepone = np.matmul(v_unit_tran, S)
    n = np.int(np.size(sigmasq_stepone[0])/2)
    sigmasq_steptwo = sigmasq_stepone.reshape((2,n),order= 'F')
    sigmasq = np.matmul(v_unit_tran, sigmasq_steptwo)
    lnL = -(0.5)*np.sum(del_array[0]**2/sigmasq[0]**2)
   
    return lnL


###############################################################################
def willywonka(Theta, dist, vel, dist_err, vel_err):

    m = Theta
    theta = np.arctan(m)
    delx = (dist - vel/m)**2/dist_err**2
    dely = (vel - dist*m)**2/vel_err**2
    lnL = - (delx*np.sin(theta)**2 + dely*np.cos(theta)**2)
    lnL = np.sum(lnL)

    return lnL

####################################################################################

def lnprior(theta):
    
    m = theta
    if 0  < m < 200.: 
        return 0.0
    return -np.inf

###################################################################################3

def lnprob(Theta, dist, vel, dist_err, vel_err):
    
    lprior = lnprior(Theta)
    if not np.isfinite(lprior):
        return -np.inf
    return lprior + willywonka(Theta, dist, vel, dist_err, vel_err)

##################################################################################

def hpd(samples, flatlnprob, mass_frac):
    max_ind = np.argmax(flatlnprob)
    mode = samples[max_ind]
    d = np.sort(np.copy(samples))
    n = len(samples)
    n_samples = np.floor(mass_frac * n).astype(int)
    int_width = d[n_samples:] - d[:n-n_samples]
    min_int = np.argmin(int_width)
    return np.array([mode, d[min_int],d[min_int+n_samples]])


####################################################################################

