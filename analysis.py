import numpy as np
import h5py
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import hconst_dict as hd
import sys
from matplotlib import rc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.gridspec import GridSpec



#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True) 
rc("text.latex", unicode = True)
rc("font", size =18., family = 'serif')

path_to_output = "/home/shivani.shah/Projects/LIGO/analysis/mcmc/runs/60_shm/0p15/output/case"
path_to_plots = "/home/shivani.shah/Projects/LIGO/analysis/mcmc/runs/60_shm/0p15/plots"


####################################################################################################3
############################################################################################3######
#To do:
#-title
#-what to write for the legend
#-errorbars or no
#############################################################################################3###

def h0_trend(path_to_output):
    cases = [3,4]#,4]#[1,2,3,4]
    dtpts = [1,2,4,10,20,30,40,50,60,80,100,120,140]
    #fig, (h0ax,intax) = plt.subplots(2,1, sharex = True,gridspec_kw = {"hspace":0})
    #fig.set_figheight(9)
    #fig.set_figwidth(5.5)
    fig2, intax = plt.subplots()
    fig, h0ax = plt.subplots()
    colors = ["red", "blue", "green", "red"]
    labels = ["case1", "case2", "Fiducial Model", "Vel Correction Model"]
    ###############################################################################################
    for case in cases:
        y = np.empty(np.size(dtpts))
        y_hi = np.empty(np.size(dtpts))
        y_lo = np.empty(np.size(dtpts))
        y_int = np.empty(np.size(dtpts))
        for i in range(len(dtpts)):
            dictionary = hd.read_output_hdf5(path_to_output+np.str(case)+"_30runs.hdf5", "dtpt"+np.str(dtpts[i]))
            samples = dictionary["samples"]; flatlnprob = dictionary["flatlnprob"]
            #samples = samples.reshape(1,-1)[0]; flatlnprob = flatlnprob.reshape(1,-1)[0]
            #mode, mode_lo,mode_hi = hd.hpd(samples, flatlnprob, 0.68)
            ntasks = 32
            runs = 3
            mode_ar = np.empty(ntasks*runs)
            mode_lo_ar = np.empty(ntasks*runs)
            mode_hi_ar = np.empty(ntasks*runs)
            fill_index = 0
            for m in range(ntasks):
                for n in range(runs):
                    sample = samples[m,n,:]
                    flatln = flatlnprob[m,n,:]
                    mode, mode_lo, mode_hi = hd.hpd(sample, flatln, 0.68)
                    mode_ar[fill_index] = mode; mode_lo_ar[fill_index] = mode_lo; mode_hi_ar[fill_index] = mode_hi
                    fill_index += 1

            #y_lo[i], y[i], y_hi[i] = np.percentile(mode_ar, [16.,50.,84.])
            y[i] = np.mean(mode_ar)#; y_hi[i] = np.mean(mode_hi_ar); y_lo[i] = np.mean(mode_lo_ar)
            y_int[i] = np.sqrt(np.var(mode_ar))
            #y[i] = mode; y_hi[i] = mode_hi; y_lo[i] = mode_lo
            #y_int[i] = y_hi[i] - y_lo[i]
        '''
        if case == 3:
            h0ax.errorbar(dtpts, y, yerr = y_int/np.sqrt(95.),marker = '.', color = colors[case-1], label = labels[case-1])
            intax.plot(dtpts, y_int, marker = 'o', linestyle = 'none', color = colors[case-1], label = labels[case-1])
        else:
            h0ax.plot(dtpts, y, linestyle = 'none')
            intax.plot(dtpts, y_int, linestyle = 'none')
        '''
        h0ax.errorbar(dtpts, y, yerr = y_int/np.sqrt(95.),marker = '.', color = colors[case-1], label = labels[case-1])
        intax.plot(dtpts, y_int, marker = 'o', linestyle = 'none', color = colors[case-1], label = labels[case-1])
        
        line_ = np.empty(len(dtpts))
        line_.fill(100.)
        h0ax.plot(dtpts, line_, linestyle = '--', color = 'k')
        
    ############################################################################################
    h0ax.set_xlabel(r"Detections")
    h0ax.set_ylabel(r"$\mathrm{H_0/h}$ "
                    r"$\mathrm{[km/s/Mpc]}$", labelpad = 15)

    #h0ax.set_ylim(bottom=80., top = 120.)
    h0ax.tick_params(direction='in',width= 2.0, size = 10)
    h0ax.tick_params(direction='in',width= 0.5, size = 5, which = 'minor')
    h0ax.yaxis.set_major_locator(MultipleLocator(0.5))
    h0ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    h0ax.set_xlim((-10,160))
    h0ax.set_ylim((99.0,101.5))
    #h0ax.set_xticks(dtpts)
    #h0ax.set_xlim(left = 0., right = 62.)
    h0ax.xaxis.set_major_locator(MultipleLocator(20))
    h0ax.xaxis.set_minor_locator(MultipleLocator(5))
    
    for axis in ["top", "bottom", "left", "right"]:
        h0ax.spines[axis].set_linewidth(2.)
    
    h0ax.margins(0.1, tight = False)
    h0ax.legend()
    fig.savefig(path_to_plots+"/30runs/h0_trend_case3&4.png", bbox_inches = "tight", dpi = 300)
    ####################################################################3333####################
    
    intax.set_xlabel(r"Detections")
    intax.set_ylabel(r"$1\sigma$ credible interval/h\ [km/s/Mpc]", labelpad = 15)

    #intax.set_ylim(bottom=80., top = 120.)
    #intax.set_xticks(dtpts)
    intax.tick_params(direction='in',width= 2.0, size = 10)
    intax.tick_params(direction='in',width= 0.5, size = 5, which = 'minor')
    intax.yaxis.set_major_locator(MultipleLocator(1))
    intax.yaxis.set_minor_locator(MultipleLocator(0.2))
    intax.set_xlim((-10,160))
    intax.set_ylim((0,6))
    #intax.set_xlim(left = 0., right = 62.)
    intax.xaxis.set_major_locator(MultipleLocator(20))
    intax.xaxis.set_minor_locator(MultipleLocator(5))
    intax.xaxis.set_ticks_position(position="default")
    for axis in ["top", "bottom", "left", "right"]:
        intax.spines[axis].set_linewidth(2.0)
    
    intax.margins(0.1, tight = False)
    intax.legend()
    fig2.savefig(path_to_plots+"/30runs/h0_int_case3&4.png", bbox_inches = 'tight', dpi = 300)
    #fig.savefig(path_to_plots+"/30runs/h0trend_case3.png", bbox_inches = "tight", dpi = 300)
    #plt.show("fig")
    
##################################################################################################333
#####################################################################################################
def pecvel_contribution(path_to_output):
    cases = [4]
    dtpts = [1,2,4,10,20,30,40,50,60]
    vel0_ar = []
    vel_err_ar = []
    vel_ar = []
    dist0_ar = []
    dist_ar = []
    for case in cases:
        for i in range(len(dtpts)):
            dictionary = hd.read_output_hdf5(path_to_output+np.str(case)+"_try3.hdf5", "dtpt"+np.str(dtpts[i]))
            vel0 = dictionary["vel0"].reshape(1,-1)[0]
            vel_err = dictionary["vel_err"].reshape(1,-1)[0]
            vel = dictionary["vel"].reshape(1,-1)[0]
            dist0 = dictionary["dist0"].reshape(1,-1)[0]
            dist = dictionary["dist"].reshape(1,-1)[0]

            if i == 0:
                vel0_ar = vel0
                vel_err_ar = vel_err
                vel_ar = vel
                dist0_ar = dist0
                dist_ar = dist
            else:
                vel0_ar = np.append(vel0_ar, vel0)
                vel_err_ar = np.append(vel_err_ar, vel_err)
                vel_ar = np.append(vel_ar, vel)
                dist0_ar = np.append(dist0_ar, dist0)
                dist_ar = np.append(dist_ar, dist)

    
    hinit = vel0_ar/dist_ar
    #hfin = vel_ar/dist_ar
    vpec_true = vel0_ar - 100.*dist0_ar
    vpec_derive = vel0_ar - vel_ar
    hfin = (vel0_ar - vpec_true)/dist_ar
    fig = plt.figure()
    gs = GridSpec(4,4)
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3], sharex = ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4,3], sharey = ax_joint)
    ax_joint.scatter(hfin,hinit)
    ax_marg_x.hist(hfin, bins = 50)
    ax_marg_y.hist(hinit, orientation = 'horizontal', bins = 50)
    x = np.arange(50., 250., 1.) 
    ax_joint.plot(x, x, color = 'k', linestyle = '--', linewidth = 1.5)
    y = np.empty(np.size(x))
    y.fill(100.)
    ax_joint.plot(x, y, color = 'k', linestyle = '-', linewidth = 1.5)
    ax_joint.plot(y,x, color = 'k', linestyle = '-', linewidth = 1.5)
    ax_joint.set_ylim(bottom = 50, top = 250.)
    ax_joint.set_xlim(left = 50., right = 250.)
    ax_marg_x.set_ylim(bottom = 0, top = 1000.)
    ax_marg_y.set_xlim(left = 0, right = 1000.)



    '''
    ax_joint.scatter(vpec_derive,vpec_true)
    ax_marg_x.hist(vpec_derive, bins = 50)
    ax_marg_y.hist(vpec_true, orientation = 'horizontal', bins = 50)
    x = np.arange(-1500., 1000., 10.) 
    ax_joint.plot(x, x, color = 'k', linestyle = '--', linewidth = 1.5)
    y = np.empty(np.size(x))
    y.fill(0.)
    #ax_joint.plot(x, y, color = 'k', linestyle = '-', linewidth = 1.5)
    #ax_joint.plot(y,x, color = 'k', linestyle = '-', linewidth = 1.5)
    
    ax_joint.set_ylim(bottom = -1500., top = 1000.)
    ax_joint.set_xlim(left = -1500., right = 1000.)
    ax_joint.yaxis.set_minor_locator(MultipleLocator(250))
    ax_joint.xaxis.set_minor_locator(MultipleLocator(250))
    ax_joint.yaxis.set_major_locator(MultipleLocator(500))
    ax_joint.xaxis.set_major_locator(MultipleLocator(500))
    #ax_marg_x.set_ylim(bottom = 0, top = 6500)
    #ax_marg_y.set_xlim(left = 0, right = 6500)
    #ax_marg_x.set_major_locator(MultipleLocator(3000))
    #ax_marg_y.set_major_locator(MultipleLocator(3000))
    '''
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_joint.set_xlabel(r"$\mathrm{H_0/h\ Final\ [km/s/Mpc]}$")
    ax_joint.set_ylabel(r"$\mathrm{H_0/h\ Initial\ [km/s/Mpc]}$")
    ax_joint.tick_params(direction='in',width= 2.0, size = 10)
    #ax_joint.tick_params(direction='in',width= 1.0, size = 4, which = 'minor')
    ax_marg_x.tick_params(direction='in', width=2.0, size = 10)
    #ax_marg_x.tick_params(direction='in', width=1.0, size = 4, which = 'minor')
    #ax_marg_y.tick_params(direction='in', width=1.0, size = 4, which = 'minor')
    ax_marg_y.tick_params(direction = 'in', width = 2.0, size =10)
    for axis in ["top", "bottom", "left", "right"]:
        ax_joint.spines[axis].set_linewidth(2.)
        ax_marg_x.spines[axis].set_linewidth(2.)
        ax_marg_y.spines[axis].set_linewidth(2.)
    plt.savefig(path_to_plots+"/H0_correlation.png", bbox_inches = 'tight')
    plt.show()
    

    sys.exit()
    #vel_ar += vel_err_ar
    hinit = vel0_ar/dist_ar
    vpec = vel0_ar - 100.*dist0_ar
    hfin = (vel0_ar - vpec)/dist_ar
    hfin2 = vel_ar/dist_ar
    res1 = np.abs(100.-hinit)
    res2 = np.abs(100.-hfin)
    res3 = (vel0_ar-vel_ar)
    res4 = np.abs(100. - (vel0_ar/dist0_ar))
   
    plt.figure()
    #plt.plot(res1,color = 'k', label = "Total l.o.s velocity")
    #plt.plot(res2, color = 'crimson', label = "Corrected l.o.s velocity")
    #plt.xlabel(r"Subhalo Number")
    #plt.ylabel(r"Residual")
    #plt.legend()
    #plt.savefig(path_to_plots+"/res_comp.png", bbox_inches = "tight", dpi = 300)
    

    plt.plot(hinit, hfin, linestyle = 'none', marker = 'o')
    plt.plot(hinit, hfin, linestyle = 'none', marker = 'o')
    #plt.ylim(50,230)
    #plt.xlim(50, 230)
    plt.xlabel("H0 init")
    #plt.xticks([-1000,-500,0,500,1000])
    #plt.yticks([-1000,-500,0,500,1000])
    plt.ylabel("H fin")
    #plt.plot(res3, color = 'green')
    #plt.plot(vpec[::50], color = 'blue')
    #plt.plot(np.abs(res3[::50]-vpec[::50]), color = 'orange')
    #plt.plot(vel_err_ar[::50], color = 'blue')
    
    plt.show()
    '''
    percent = vel_err_ar/vel0_ar
    ind = [percent > 0.10]
    percent_subset = percent[ind]
    print(np.float(np.size(percent_subset))/np.float(np.size(percent)))
    print(np.size(percent))
    #plt.hist(pec_fraction, bins = 10)
    #plt.show()
    '''

######################################################################################################
######################################################################################################

def plot_test():
    x = [1,2,4,10,20,30,40,50,60]
    y = [120.,80.,105., 98., 102., 100., 101.5, 100.5, 100.2]
    fig, h0ax = plt.subplots()
    h0ax.plot(x,y, marker = 'o', color = 'crimson', markersize = 5, linewidth = 1.5, label = 'test')
    h0ax.set_xlabel(r"Detections")
    h0ax.set_ylabel(r"H0")
    #y_limits = h0ax.get_yticks()
    #y_range = np.arange(y_limits[0], y_limits[-1], 1.)
    h0ax.yaxis.set_major_locator(MultipleLocator(5))
    h0ax.yaxis.set_minor_locator(MultipleLocator(1))
    h0ax.set_xticks(x)
    for axis in ["top", "bottom", "left", "right"]:
        h0ax.spines[axis].set_linewidth(2.0)
    h0ax.tick_params(direction='in',width= 2.0, size = 8)
    h0ax.tick_params(direction='in',width= 0.5, size = 3, which = 'minor')


    h0ax.legend()
    plt.show()

######################################################################################################
#####################################################################################################

def mcmc_trend():
    cases = [3]#,2,3,4]
    dtpts = [1,2,4,10,40,60,100,140]
    for i in range(len(dtpts)):
        for case in cases:
            dictionary = hd.read_output_hdf5(path_to_output+np.str(case)+"_30runs.hdf5", "dtpt"+np.str(dtpts[i]))
            dist = dictionary["dist"].reshape(1,-1)[0]; vel = dictionary["vel"].reshape(1,-1)[0]; samples = dictionary["samples"]; flatlnprob = dictionary["flatlnprob"]
            ntasks = 32
            runs = 3
            mode_ar = np.empty(ntasks*runs)
            mode_lo_ar = np.empty(ntasks*runs)
            mode_hi_ar = np.empty(ntasks*runs)
            fill_index = 0
            for m in range(ntasks):
                for n in range(runs):
                    sample = samples[m,n,:]
                    flatln = flatlnprob[m,n,:]
                    mode, mode_lo, mode_hi = hd.hpd(sample, flatln, 0.68)
                    mode_ar[fill_index] = mode; mode_lo_ar[fill_index] = mode_lo; mode_hi_ar[fill_index] = mode_hi
                    fill_index += 1
            samples_all = samples.reshape(1,-1)[0]
            flatlnprob_all = flatlnprob.reshape(1,-1)[0]
            #mode_av, mode_lo_av, mode_hi_av = hd.hpd(samples_all, flatlnprob_all, 0.68)
            #mode_av, mode_lo_av, mode_hi_av = np.percentile(mode_ar, [16.,50.,84.])
            mode_av = np.mean(mode_ar)#; mode_lo_av = np.mean(mode_lo_ar); mode_hi_av = np.mean(mode_hi_ar)
            print(np.size(mode_ar))
            
            ####################################################################################33
            dist_ran = np.arange(0.,100.,1.)
            fig, (ax,ax2) = plt.subplots(1,2)
            fig.set_figheight(5)
            fig.set_figwidth(12)
            if dtpts[i] == 10: 
                s = 2.5 
            elif dtpts[i] == 20 or dtpts[i] == 30:
                s = 1.5
            elif dtpts[i] == 40 or dtpts[i] == 50 or dtpts[i] == 60:
                s = 1
            else: s=5
            for slope in mode_ar:
                ax.plot(dist_ran, dist_ran*slope, color = 'navajowhite', linestyle = '-')
            ax.plot(dist, vel,linestyle='none', marker = '.', color = "orange", label = "Detections", markersize = s)
            ax.plot(dist_ran, dist_ran*mode_av, color = 'darkorange', linestyle = '-', label = 'Average of MCMC Fits', linewidth = 3)
            ax.plot(dist_ran, dist_ran*100., color = 'purple', label = 'True Fit', linewidth = 3, linestyle = '--')
            ######################################################################################
            ax.set_xlabel(r"Distance/h [Mpc]")
            ax.set_ylabel(r"Hubble Velocity [km/s]", labelpad = 5)
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(2.)
            ax.tick_params(direction='in',width= 2.0, size = 10)
            ax.tick_params(direction='in',width= 0.5, size = 5, which = 'minor')
            ax.set_ylim(bottom = -750.,top=10500.)
            ax.set_xlim(left = -5., right=105.)
            ax.xaxis.set_minor_locator(MultipleLocator(2))
            ax.yaxis.set_minor_locator(MultipleLocator(250))
            ax.margins(0.1, tight = False)
            #########################################################################################

            '''
            ax2 = plt.axes([0,0,1,1])
            ip = InsetPosition(ax, [0.7,0.1,0.2,0.2])
            ax2.set_axes_locator(ip)
            dist_newran = np.arange(30., 40., 1)
            ax2.plot(dist_newran, dist_newran*mode_av, linestyle = '-', linewidth = 3, color = 'darkorange')
            ax2.plot(dist_newran, dist_newran*100., linestyle = '--', linewidth = 3, color = 'purple')
            for axis in ["top", "bottom", "left", "right"]:
                ax2.spines[axis].set_linewidth(1.5)
            ax2.tick_params(direction='in',width= 1.5, size = 9, labelsize = 12.)
            #ax2.tick_params(direction='in',width= 0.5, size = 4, which = 'minor')
            ax2.set_xlim(left = 30, right = 37)
            ax2.xaxis.set_major_locator(MultipleLocator(4))
            ax2.set_ylim(top=3700, bottom= 3000)
            ax2.yaxis.set_major_locator(MultipleLocator(400))
            '''

            ax.legend(loc=2)
            #ax.set_title(np.str(dtpts[i])+" detection(s) of Case "+np.str(case))
            ax.set_title(np.str(dtpts[i]) + " detection(s)")
            ax2.hist(mode_ar, color = "darkorange", density =  True,bins=20)
            ax2.set_xlim(left = 70., right = 130.)
            for axis in ["top", "bottom", "left", "right"]:
                ax2.spines[axis].set_linewidth(2.)
            ax2.tick_params(direction='in',width= 2.0, size = 10)
            ax2.tick_params(direction='in',width= 0.5, size = 5, which = 'minor')
            ax2.set_xlabel(r"H$_0$/h km/s/Mpc")
            ax2.set_ylabel(r"PDF")
            ax2.xaxis.set_minor_locator(MultipleLocator(2))


            plt.savefig(path_to_plots + "/30runs/mcmc_trend/dtpt" + np.str(dtpts[i])+"_"+np.str(case)+".png", bbox_inches = "tight", dpi=300)
            #plt.close()
            #plt.show()
            

            









print("here")
#h0_trend(path_to_output)
#pecvel_contribution(path_to_output)
#plot_test()
mcmc_trend()
