import argparse
import os
import numpy as np
import pandas as pd
from astropy.table import Table

def parse_arguments():
    parser = argparse.ArgumentParser(description="Exocomet Injection Testing")
    parser.add_argument("path", help="Target directory", default=".")
    parser.add_argument("-s", help="TESS sector", dest="sector", default=6, type=int)
    parser.add_argument("-n", help="Number of lightcurves to test", dest="number", 
                       default=2000, type=int)
    parser.add_argument("-mag_lower", type=int, help="Lower magnitude limit")
    parser.add_argument("-mag_higher", type=int, help="Upper magnitude limit")
    return parser.parse_args()

def select_lightcurves(lookup_data, mag_lower, mag_higher, n_samples):
    """Select lightcurves within specified magnitude range"""
    return lookup_data.Filename[
        (lookup_data.Magnitude >= mag_lower) & 
        (lookup_data.Magnitude <= mag_higher)
    ].sample(n_samples, replace=True).values

def inject_comet_signal(time, flux, depth, event_time):
    """Inject artificial comet signal into lightcurve"""
    # Using fixed parameters for comet model
    tau_1 = 0.303  # Rise time
    tau_2 = 0.340  # Decay time
    
    # Simple asymmetric dip model
    dt = time - event_time
    comet = np.where(dt <= 0, 
                     np.exp(dt/tau_1),
                     np.exp(-dt/tau_2))
    return flux * (1 - depth * comet)

def process_lightcurve(filename, sector, depth, event_time):
    """Process single lightcurve with injection"""
    # Load lightcurve
    data = Table.read(filename)
    time = data['time']
    flux = data['flux']
    
    # Inject comet signal
    injected_flux = inject_comet_signal(time, flux, depth, event_time)
    
    # Add to dataframe
    data['injected_flux'] = injected_flux
    
    return data

def run_injection_recovery(args):
    # Load sector lookup table
    lookup = pd.read_csv(f"sector{args.sector}lookup.csv")
    
    # Select target lightcurves
    target_files = select_lightcurves(
        lookup, args.mag_lower, args.mag_higher, args.number
    )
    
    results = []
    for filename in target_files:
        # Generate random injection parameters
        depth = 10 ** np.random.uniform(-4, -2)
        event_time = np.random.uniform(1470, 1489)  # Typical TESS sector timespan
        
        # Process lightcurve
        processed_data = process_lightcurve(filename, args.sector, depth, event_time)
        
        # Store results
        results.append({
            'filename': filename,
            'injected_depth': depth,
            'injected_time': event_time,
            'magnitude': lookup[lookup.Filename == filename].Magnitude.values[0]
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'injection_results_sector{args.sector}.csv', index=False)
    
    return results_df

if __name__ == "__main__":
    args = parse_arguments()
    results = run_injection_recovery(args)