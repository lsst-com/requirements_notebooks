import numpy as np

__all__ = ["select_stars"]

def select_stars(dataset, calib, has_cmodel=True):
    """
    Select all sources with
    base_ClassificationExtendedness_value == 0
    & base_PsfFlux_flag == 0
    & np.abs(PsfMag-CModelMag)<0.03
    & PsfMag<25.0
    
    Parameters
    ----------
    dataset -- a src or deepCoadd_forcedSrc catalog
    calib -- the photometric calibration object needed to convert fluxes into mags
    
    Returns
    -------
    numpy array of PSF magnitudes
    numpy array of cModel magnitudes
    numpy array of booleans set to 'True' for everything that meets the criteria above
    """
    extendedness_flag = dataset['base_ClassificationExtendedness_value']==0
    well_measured_flag = dataset['base_PsfFlux_flag']==0
    mag_and_error = calib.instFluxToMagnitude(dataset, 'base_PsfFlux')
    mag = mag_and_error[:,0]
    mag_error = mag_and_error[:,1]
    colnames = dataset.getSchema().getNames()
    if 'modelfit_CModel_instFlux' in colnames:
        model_mag = calib.instFluxToMagnitude(dataset, 'modelfit_CModel')[:,0]
        mag_m_model_flag = np.abs(mag-model_mag)<0.03
    elif not has_cmodel:
        model_mag = np.NaN*np.ones(len(mag), dtype=float)
        mag_m_model_flag = np.ones(len(mag), dtype=bool)
    else:
        raise RuntimeError("Does not contain cModel")
    brightness_flag = mag<25.0
    brightness_finite = np.isfinite(mag)
    flag = extendedness_flag & well_measured_flag & mag_m_model_flag & brightness_finite & brightness_flag
    return mag, mag_error, model_mag, flag
