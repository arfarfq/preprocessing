import numpy as np
import lightkurve as lk
from tqdm import tqdm
import pandas as pd
import sqlite3
from scipy import interpolate
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.time import TimeDelta
import h5py
import concurrent.futures
from scipy.signal import medfilt
import lightkurve as lk
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline


# Global configuration
TEST_MODE = True
TEST_LIMIT = 4 if TEST_MODE else None
MAX_WORKERS = 64
SAMPLE_SIZE= 256 * 4
BIN_SIZE_DENOMINATOR = 10
SAVITZKY_GOLAY_WINDOW = 21
SAVITZKY_GOLAY_POLYORDER = 2


def fetchTIC():
    # Connect to database
    connection = sqlite3.connect('/mnt/data/toi_database_no_folder.db') 

    df_path = pd.read_sql_query("SELECT TIC, Sector, path_to_fits FROM LightCurves", connection)
    df_feat = pd.read_sql_query('SELECT TIC, "TOI Disposition" FROM TOIs', connection)
    connection.close()

    return df_path, df_feat

def interpolate_lcs(lc):
    time_lc = lc.time.value
    flux_lc = lc.flux.value

    mask = np.isnan(flux_lc) 
    if np.any(mask):
        f_interp = interpolate.interp1d(time_lc[~mask], flux_lc[~mask], 
                                        kind='linear', bounds_error=False, fill_value="extrapolate")
        flux_lc = f_interp(time_lc)  # Interpolated flux
        # Update the entire flux array, preserving units
        lc = lk.LightCurve(time=lc.time, flux=flux_lc * lc.flux.unit)

    return lc


def lc_preprocess(lc):

    # Preprocessing steps
    lc = lc.remove_nans().remove_outliers(sigma_upper=5, sigma_lower=np.inf).normalize()
    lc_flat = lc.flatten(window_length=1001, polyorder=2)

    # BLS periodogram
    period = np.linspace(1, 20, 10000)
    pg = lc_flat.to_periodogram(method='bls', period=period, frequency_factor=500)
    best_period = pg.period_at_max_power
    t0 = pg.transit_time_at_max_power
    t_dur = pg.duration_at_max_power
    

    # Savitzky Golay
    filtered_flux = savgol_filter(lc_flat.flux.value, window_length= SAVITZKY_GOLAY_WINDOW, polyorder=SAVITZKY_GOLAY_POLYORDER)
    svg = lk.LightCurve(time=lc_flat.time, flux=filtered_flux)

    # Fold on best period
    global_lc = svg.fold(period=best_period, epoch_time=t0)

    global_lc_no_filter = lc_flat.fold(period=best_period, epoch_time=t0)


    # Global binning (1/10th of transit duration)
    global_bin_size = t_dur / BIN_SIZE_DENOMINATOR
    global_lc_binned = global_lc.bin(time_bin_size=global_bin_size)
    global_lc_binned = interpolate_lcs(global_lc_binned)
    
    # Local Light Curce
    time_limit_phase = 1.5 * t_dur
    local_mask = (global_lc.time >= -time_limit_phase) & \
                 (global_lc.time <= time_limit_phase)
    local_lc = global_lc[local_mask]
    

    if len(local_lc.time) == 0:
        raise ValueError("Error: local_lc is empty after phase selection.")
    
    # Binning Local Light Curve
    target_local_bin_size = t_dur / BIN_SIZE_DENOMINATOR /2
    cadence = np.median(np.diff(local_lc.time)).to_value('d') * u.day
    final_local_bin_size = max(target_local_bin_size, cadence)

    # Now use final_local_bin_size_td in  binning
    local_lc_binned = local_lc.bin(time_bin_size=final_local_bin_size)
    local_lc_binned = interpolate_lcs(local_lc_binned)
    
    return local_lc_binned, global_lc_binned

def resample_lightcurve(lc, length: int = SAMPLE_SIZE):
    """
    Resamples a Lightkurve LightCurve object to a fixed number of time steps.

    Parameters:
    lc (LightCurve): Lightkurve LightCurve object.
    length (int): Desired output length (number of time steps).

    Returns:
    tuple: (resampled_time, resampled_flux), both as numpy arrays.
    """
    # Convert time to NumPy float array 
    time = lc.time.value  
    flux = lc.flux

    # Remove NaNs
    mask = ~np.isnan(flux)
    time, flux = time[mask], flux[mask]

    # Interpolate to fixed length
    interp_func = interp1d(time, flux, kind="linear", fill_value="extrapolate")
    resampled_time = np.linspace(time.min(), time.max(), length)
    resampled_flux = interp_func(resampled_time)

    return resampled_flux.reshape((length, 1)).astype(np.float32)


def label_toi_disposition(disposition):
    positive_labels = ['PC', 'KP', 'CP']
    disposition = str(disposition).strip().upper()  # Normalise
    if pd.isna(disposition) or not disposition:  # Handle NaN or empty
        return 0
    return 1 if disposition in positive_labels else 0

def process_single_light_curve(row):
    """Process a single light curve row and return results."""
    tic, path, disposition = row['TIC'], row['path_to_fits'], row['TOI Disposition']
    
    try:
        lc = lk.read(path)
        local_lc, global_lc = lc_preprocess(lc)
        
        local_input = resample_lightcurve(local_lc, length=SAMPLE_SIZE)
        global_input = resample_lightcurve(global_lc, length=SAMPLE_SIZE)
        label = label_toi_disposition(disposition)
        
        if local_input is None or len(local_input) == 0:
            local_input = np.zeros((SAMPLE_SIZE, 1), dtype=np.float32)
            
        if global_input is None or len(global_input) == 0:
            return None
            
        return (local_input, global_input, label)
    
    except Exception as e:
        return None

def process_light_curves(df_path, df_feat, test_limit=None):
    """Process light curves in parallel and assign labels based on TOI Disposition."""
    local_inputs, global_inputs, labels = [], [], []
    
    # Prepare the data frame 
    df_feat = df_feat.drop_duplicates(subset='TIC', keep='first')
    df_merged = df_path.merge(df_feat, on='TIC', how='left')
    df_merged['TOI Disposition'] = df_merged['TOI Disposition'].fillna('UNK')
    df_merged['label'] = df_merged['TOI Disposition'].apply(label_toi_disposition)

    if TEST_MODE and test_limit:
        df_pos = df_merged[df_merged['label'] == 1]
        df_neg = df_merged[df_merged['label'] == 0]
        min_samples = min(len(df_pos), len(df_neg), test_limit // 2)
        df_sampled = pd.concat([df_pos.sample(min_samples, random_state=42),
                              df_neg.sample(min_samples, random_state=42)])
        df_merged = df_sampled.sample(frac=1, random_state=42)

    # Parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_single_light_curve, row): row 
                        for _, row in df_merged.iterrows()}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_row), 
                         total=len(future_to_row), 
                         desc="Processing Light Curves"):
            result = future.result()
            if result is not None:
                local_input, global_input, label = result
                local_inputs.append(local_input)
                global_inputs.append(global_input)
                labels.append(label)

    return (np.array(local_inputs), 
            np.array(global_inputs), 
            np.array(labels), 
            df_merged)

def save_processed_data(local_inputs, global_inputs, labels, df_merged, filename="/mnt/data/LCs_1024_CNN_Input.h5"):
    """Save processed light curves and metadata to an HDF5 file."""

    
    with h5py.File(filename, "w") as f:
        # Save numpy arrays
        f.create_dataset("local_inputs", data=local_inputs, compression="gzip", chunks=True)
        f.create_dataset("global_inputs", data=global_inputs, compression="gzip", chunks=True)
        f.create_dataset("labels", data=labels.astype(int), compression="gzip")
        
        # Define metadata dtype with EXACT column names matching df_merged
        metadata_dtype = [
            ('TIC', int),
            ('sector', int),  # <-- Now matches DataFrame's 'Sector' (capitalized)
            ('path_to_fits', h5py.string_dtype()),
            ('TOI Disposition', h5py.string_dtype()),
            ('label', int)
        ]
        
        # Create structured array
        metadata_arr = np.zeros(len(df_merged), dtype=metadata_dtype)
        for col in df_merged.columns:
            if col in ['TIC', 'sector', 'label']:  # Columns to save as integers
                metadata_arr[col] = df_merged[col].values.astype(int)
            else:  # Columns to save as strings
                metadata_arr[col] = df_merged[col].values.astype(str)
        
        f.create_dataset("metadata", data=metadata_arr, compression="gzip")

if __name__ == '__main__':
    # Main execution block
    df_path, df_feat = fetchTIC()

    local_inputs, global_inputs, labels, df_merged = process_light_curves(df_path, df_feat, TEST_LIMIT)

    save_processed_data(local_inputs=local_inputs, global_inputs=global_inputs, labels=labels, df_merged=df_merged)
