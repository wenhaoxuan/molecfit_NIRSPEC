import astropy.io.fits as fits
import astropy.io.ascii as ascii
from astropy.table import Table, Column, MaskedColumn
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyAstronomy.pyasl as pyastro
from scipy import interpolate
from scipy.stats import sigmaclip
import csv

## To load pynirspec wavelength-calibrated spectra into fits format for molecfit
target_name = 'DHTaub'
set_mode = True # leave-one-out-sets
manual_offset = False

print ('manual offset?')
print manual_offset

if manual_offset:
    append_text = '-MO'
else:
    append_text = ''

if not set_mode:
    for onum in ['1','2','3','4','5','6']:
        spec = '../pynirspec-out/2017_11_03/'+target_name+'/WAVECAL_SPECTRA/order'+onum+'-calibrated-spectra'+append_text+'.p'
        order = pickle.load(open(spec,'rb'))

        ## Default: no trimming
        l_trim_size = 0
        r_trim_size = -len(order['fluxpos'])

        if target_name == 'DHTau' or target_name == 'HIP8535':
            l_trim_size = 18
            r_trim_size = 18
        flux_pos = np.array(order['fluxpos'])[l_trim_size:-r_trim_size]
        uflux_pos = np.array(order['ufluxpos'])[l_trim_size:-r_trim_size]
        wave_pos = np.array(order['wavepos'])[l_trim_size:-r_trim_size]
        flux_neg = np.array(order['fluxneg'])[l_trim_size:-r_trim_size]
        uflux_neg = np.array(order['ufluxneg'])[l_trim_size:-r_trim_size]
        wave_neg = np.array(order['waveneg'])[l_trim_size:-r_trim_size]

        #print flux_pos
        #print uflux_pos

        print len(flux_pos), len(wave_pos), len(uflux_pos)

        #print len(flux_pos), len(wave_pos)
        order_pos = np.core.records.fromarrays([wave_pos,flux_pos,uflux_pos],names='Wavelength,Extracted_OPT,Error_OPT')
        order_neg = np.core.records.fromarrays([wave_neg,flux_neg,uflux_neg],names='Wavelength,Extracted_OPT,Error_OPT')

        fits.writeto('./share/molecfit/examples/NIRSPEC/'+target_name+'_o'+onum+'_pos'+append_text+'.fits', order_pos, header=None, overwrite=True)
        fits.writeto('./share/molecfit/examples/NIRSPEC/'+target_name+'_o'+onum+'_neg'+append_text+'.fits', order_neg, header=None, overwrite=True)

else:
    for i in range(8):
        for onum in ['1','2','3','4','5','6']:
            spec = '../pynirspec-out/2017_11_03/'+target_name+'-set'+str(i+1)+'/WAVECAL_SPECTRA/order'+onum+'-calibrated-spectra'+append_text+'.p'
            order = pickle.load(open(spec,'rb'))

            ## Default: no trimming
            l_trim_size = 0
            r_trim_size = -len(order['fluxpos'])

            if target_name == 'DHTau' or target_name == 'HIP8535':
                l_trim_size = 18
                r_trim_size = 18
            flux_pos = np.array(order['fluxpos'])[l_trim_size:-r_trim_size]
            uflux_pos = np.array(order['ufluxpos'])[l_trim_size:-r_trim_size]
            wave_pos = np.array(order['wavepos'])[l_trim_size:-r_trim_size]
            flux_neg = np.array(order['fluxneg'])[l_trim_size:-r_trim_size]
            uflux_neg = np.array(order['ufluxneg'])[l_trim_size:-r_trim_size]
            wave_neg = np.array(order['waveneg'])[l_trim_size:-r_trim_size]

            #print len(flux_pos), len(wave_pos), len(uflux_pos)

            order_pos = np.core.records.fromarrays([wave_pos,flux_pos,uflux_pos],names='Wavelength,Extracted_OPT,Error_OPT')
            order_neg = np.core.records.fromarrays([wave_neg,flux_neg,uflux_neg],names='Wavelength,Extracted_OPT,Error_OPT')

            fits.writeto('./share/molecfit/examples/NIRSPEC/'+target_name+'_set'+str(i+1)+'_o'+onum+'_pos'+append_text+'.fits', order_pos, header=None, overwrite=True)
            fits.writeto('./share/molecfit/examples/NIRSPEC/'+target_name+'_set'+str(i+1)+'_o'+onum+'_neg'+append_text+'.fits', order_neg, header=None, overwrite=True)


## Function make planet/stellar model, used in plots
def broad_spec(resolution, target_name = 'DHTaub'):
    if target_name == 'DHTaub':

        pl_model = pd.read_csv('../pynirspec-pipeline/Oph90_RoughModel_SolarCtoO.csv')
        flux_solar = pl_model.iloc[:, 0]
        wave_solar = pl_model.iloc[:, 1]
    elif target_name == 'DHTau':
        star_model = pd.read_csv('../pynirspec-pipeline/DHTau-Kband-PHOENIX-model.csv')
        flux_solar = star_model.iloc[:, 1]  # flux is assumed to be already normalized by median
        wave_solar = star_model.iloc[:, 0] / 10000  # convert angstroms to um

    elif target_name == 'HIP8535':
        star_model = pd.read_csv('../pynirspec-pipeline/HIP8535-Kband-PHOENIX-model.csv')
        flux_solar = star_model.iloc[:, 1]  # flux is assumed to be already normalized by median
        wave_solar = star_model.iloc[:, 0] / 10000  # convert angstroms to um

    wave_solarn = np.zeros(len(wave_solar))
    flux_solarn = np.zeros(len(flux_solar))
    for k in xrange(len(wave_solar)):
        wave_solarn[k] = wave_solar[len(wave_solar) - k - 1]
        flux_solarn[k] = flux_solar[len(flux_solar) - k - 1]

    wave_solar = wave_solarn
    flux_solar = flux_solarn
    max_wl_solar = np.max(wave_solar)
    min_wl_solar = np.min(wave_solar)

    # now interpolate the model spectrum onto an evenly spaced wavelength grid
    step = (1. / len(wave_solar))
    wave_new_solar = np.arange(min_wl_solar, max_wl_solar, step)
    f9 = interpolate.interp1d(wave_solar, flux_solar, kind='slinear')
    flux_solar_new = f9(wave_new_solar)

    # flatten stellar model
    Pixels = np.arange(0,len(flux_solar_new))
    coefs = np.polyfit(Pixels, flux_solar_new, 4)
    func = np.poly1d(coefs)
    baseline = func(Pixels)

    # Flatten spectra with a 4-order polynomial
    flux_solar_new = flux_solar_new / baseline

    broad = pyastro.instrBroadGaussFast(wave_new_solar, flux_solar_new, resolution, edgeHandling=None)
    return wave_new_solar, broad

## Function to remove deep telluric lines with sigma-clipping
def remove_deep_tel(tel_model, broadmod, input_tel_cor_data, wave_data, target_name='DHTaub'):
    if target_name == 'DHTaub':
        low_thres_t = 2.7
        high_thres_t = 3.5
    elif target_name == 'DHTau':
        low_thres = 4.0
        high_thres = 4.0
    res_spec = tel_model - broadmod  # model telluric - model spectra
    trash, low, high = sigmaclip(res_spec, low_thres_t, high_thres_t)
    print ('sigmaclip thres for telluric:')
    print high
    print low

    low_inds = np.where(res_spec < low)[0]

    # remove adjacent points to the spikes in last iteration
    r_low_inds = low_inds + 1
    l_low_inds = low_inds - 1
    low_inds = reduce(np.union1d, (low_inds, r_low_inds, r_low_inds +1, l_low_inds, l_low_inds-1))
    low_inds = low_inds[(low_inds >= 0) & (low_inds < 988)
    
    return low_inds

## Plot the telluric-corrected spectra from molecfit
# The telluric-corrected spectra is contained in the '_TAC.fits' files
# Relevant columns are:
wave = data['Wavelength']
input_data = data['Extracted_OPT'] # original input spectra
tel_cor_data = data['tacflux'] / np.median(data['tacflux']) # telluric corrected spectra
