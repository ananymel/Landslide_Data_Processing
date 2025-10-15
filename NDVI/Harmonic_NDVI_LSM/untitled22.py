#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 01:29:45 2025

@author: melis
"""

import pandas as pd
import numpy as np
from numpy.linalg import lstsq

# === Load your Excel file ===
# Replace with actual path
df = pd.read_excel(r"/Users/melis/Desktop/NDVI_Download_09_22_2025/ndvi_data_2017_cropped/NDVI_2017_AllDays.xlsx")

# Extract NDVI time series: columns 0 to 364
ndvi_cols = [str(i) for i in range(365)]
ndvi_vals = df[ndvi_cols].to_numpy(dtype=np.float32)  # shape (n_pixels, 365)

# Create the harmonic design matrix (same for all pixels)
t = np.arange(365)  # day of year
X = np.column_stack([
    np.ones_like(t),
    np.cos(2*np.pi*t/365),
    np.sin(2*np.pi*t/365),
    np.cos(4*np.pi*t/365),
    np.sin(4*np.pi*t/365)
])  # shape (365, 5)

# === Fit harmonic model to each pixel ===
n_pix = ndvi_vals.shape[0]
coefs = np.zeros((n_pix, 5), dtype=np.float32)

for i in range(n_pix):
    y = ndvi_vals[i]
    if np.isnan(y).sum() > 50:  # skip if too many missing values
        coefs[i] = np.nan
        continue
    mask = ~np.isnan(y)
    beta, *_ = lstsq(X[mask], y[mask], rcond=None)
    coefs[i] = beta  # [c0, a1, b1, a2, b2]

# === Optional: convert to amplitude/phase ===
a1, b1 = coefs[:, 1], coefs[:, 2]
amp1 = np.hypot(a1, b1)
phase1 = np.arctan2(-b1, a1)  # peak timing in radians
peak_day = (phase1 % (2*np.pi)) / (2*np.pi) * 365

# === Add to dataframe ===
df["c0"] = coefs[:, 0]
df["a1"] = a1
df["b1"] = b1
df["a2"] = coefs[:, 3]
df["b2"] = coefs[:, 4]
df["Amplitude1"] = amp1
df["Peak_DOY"] = peak_day

# === Save to CSV or Excel ===
df.to_csv("ndvi_harmonic_coefficients_2017.csv", index=False)


import matplotlib.pyplot as plt
import numpy as np

# === Setup ===
pix = 123  # choose a valid row index for your pixel
t = np.arange(365)  # day-of-year axis (0 to 364)
ndvi = ndvi_vals[pix]  # shape: (365,)
coeffs = B[pix]        # shape: (5,) = [c0, a1, b1, a2, b2]

# === Generate Harmonic Fit ===
# X already defined as (365, 5)
fit = X @ coeffs

# === Plot ===
plt.figure(figsize=(10, 4))
plt.plot(t, ndvi, 'o', markersize=3, label='Observed NDVI (2017)', alpha=0.6)
plt.plot(t, fit, 'r-', label='Harmonic Fit', linewidth=2)
plt.title(f'Pixel {pix} – Daily NDVI and Harmonic Model (2017)')
plt.xlabel('Day of Year (t)')
plt.ylabel('NDVI')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Optional: Print harmonic info ===
A1 = np.hypot(coeffs[1], coeffs[2])
peak_day = (np.arctan2(-coeffs[2], coeffs[1]) % (2*np.pi)) / (2*np.pi) * 365
print(f"A1 (amplitude of annual cycle): {A1:.4f}")
print(f"Peak NDVI day of year: {peak_day:.2f}")


#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 01:29:45 2025
@author: melis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# === Load Excel NDVI file ===
df = pd.read_excel(
    r"/Users/melis/Desktop/NDVI_Download_09_22_2025/ndvi_data_2017_cropped/NDVI_2017_AllDays.xlsx"
)

# === Extract daily NDVI values (columns 0 to 364) ===
ndvi_cols = [str(i) for i in range(365)]
ndvi_vals = df[ndvi_cols].to_numpy(dtype=np.float32)  # shape: (n_pixels, 365)

# === Create harmonic design matrix (same for all pixels) ===
t = np.arange(365)  # days 0 to 364
X = np.column_stack([
    np.ones_like(t),
    np.cos(2 * np.pi * t / 365),
    np.sin(2 * np.pi * t / 365),
    np.cos(4 * np.pi * t / 365),
    np.sin(4 * np.pi * t / 365)
])  # shape: (365, 5)

# === Fit harmonic model to each pixel ===
n_pix = ndvi_vals.shape[0]
B = np.zeros((n_pix, 5), dtype=np.float32)  # matrix of harmonic coefficients

for i in range(n_pix):
    y = ndvi_vals[i]
    if np.isnan(y).sum() > 50:
        B[i] = np.nan
        continue
    mask = ~np.isnan(y)
    beta, *_ = lstsq(X[mask], y[mask], rcond=None)
    B[i] = beta  # Store [c0, a1, b1, a2, b2]

# === Extract harmonic components ===
c0 = B[:, 0]
a1 = B[:, 1]
b1 = B[:, 2]
a2 = B[:, 3]
b2 = B[:, 4]

# === Compute amplitude and peak timing ===
amp1 = np.hypot(a1, b1)
phase1 = np.arctan2(-b1, a1)
peak_day = (phase1 % (2 * np.pi)) / (2 * np.pi) * 365

# === Save to dataframe ===
df["c0"] = c0
df["a1"] = a1
df["b1"] = b1
df["a2"] = a2
df["b2"] = b2
df["Amplitude1"] = amp1
df["Peak_DOY"] = peak_day

# === Save to CSV ===
df.to_csv("ndvi_harmonic_coefficients_2017.csv", index=False)

# === OPTIONAL: Plot NDVI vs Harmonic Fit for a chosen pixel ===
pix = 100  # Change this to a valid pixel index
y = ndvi_vals[pix]
fit = X @ B[pix]

plt.figure(figsize=(10, 4))
plt.plot(t, y, 'o', markersize=3, label='Observed NDVI (2017)', alpha=0.6)
plt.plot(t, fit, 'r-', label='Harmonic Fit', linewidth=2)
plt.title(f'Pixel {pix} – NDVI and Harmonic Model (2017)')
plt.xlabel('Day of Year')
plt.ylabel('NDVI')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Print info for selected pixel ===
print(f"Pixel {pix}")
print(f"Amplitude of annual cycle: {amp1[pix]:.4f}")
print(f"Peak NDVI Day of Year: {peak_day[pix]:.2f}")
print(f"Coefficients: c0 = {c0[pix]:.4f}, a1 = {a1[pix]:.4f}, b1 = {b1[pix]:.4f}, a2 = {a2[pix]:.4f}, b2 = {b2[pix]:.4f}")

