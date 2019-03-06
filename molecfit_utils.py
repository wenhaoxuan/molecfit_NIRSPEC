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


## Plot the telluric-corrected spectra from molecfit

# The telluric-corrected spectra is contained in the '_TAC.fits' file, 
# generated by calctrans or corrfilelist

# Relevant columns are:
wave = data['Wavelength']
input_data = data['Extracted_OPT'] # original input spectra
tel_cor_data = data['tacflux'] / np.median(data['tacflux']) # telluric corrected spectra
