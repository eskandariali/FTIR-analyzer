import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Configuration & Layout ---
st.set_page_config(page_title="FTIR Spectrum Analyzer", layout="wide")

st.title("🔬 FTIR Spectrum Analyzer & Interpreter")
st.markdown("""
This tool normalizes FTIR data, identifies peaks, and attempts to match them 
to common functional groups based on standard IR spectroscopy tables.
""")

# --- 1. Knowledge Base: FTIR Frequency Table (Updated from uploaded DB) ---
FUNCTIONAL_GROUPS = [
    # O-H / N-H Region
    {"min": 3584, "max": 3700, "group": "O-H stretching (free)", "desc": "Alcohol/Phenol (Strong, Sharp)"},
    {"min": 3200, "max": 3550, "group": "O-H stretching (H-bonded)", "desc": "Alcohol (Strong, Broad)"},
    {"min": 3490, "max": 3510, "group": "N-H stretching", "desc": "Primary Amine (Medium, Sharp)"},
    {"min": 3300, "max": 3400, "group": "N-H stretching", "desc": "Aliphatic Primary Amine (Medium, Sharp)"},
    {"min": 3310, "max": 3350, "group": "N-H stretching", "desc": "Secondary Amine (Medium, Sharp)"},
    {"min": 2800, "max": 3000, "group": "N-H stretching", "desc": "Amine Salt (Strong, Broad)"},
    {"min": 2500, "max": 3300, "group": "O-H stretching", "desc": "Carboxylic Acid (Very Strong, Very Broad)"},

    # C-H Region
    {"min": 3267, "max": 3333, "group": "C-H stretching", "desc": "Alkyne (Strong, Sharp)"},
    {"min": 3290, "max": 3310, "group": "≡C–H stretching", "desc": "Terminal Alkyne (Strong, Sharp)"},
    {"min": 3050, "max": 3100, "group": "C-H stretching", "desc": "Aromatic (Medium)"},
    {"min": 3000, "max": 3100, "group": "C-H stretching", "desc": "Alkene (Medium)"},
    {"min": 2840, "max": 3000, "group": "C-H stretching", "desc": "Alkane (Medium)"},
    {"min": 2695, "max": 2830, "group": "C-H stretching", "desc": "Aldehyde (Medium, Fermi doublet)"},
    {"min": 2550, "max": 2600, "group": "S-H stretching", "desc": "Thiol (Weak)"},

    # Triple Bond Region
    {"min": 2210, "max": 2260, "group": "C≡N stretching", "desc": "Nitrile (Medium)"},
    {"min": 2100, "max": 2260, "group": "C≡C stretching", "desc": "Alkyne (Variable)"},

    # Double Bond / Carbonyl Region
    {"min": 1760, "max": 1690, "group": "C=O stretching", "desc": "Carboxylic Acid (Strong)"},
    {"min": 1750, "max": 1735, "group": "C=O stretching", "desc": "Esters (Strong)"},
    {"min": 1740, "max": 1720, "group": "C=O stretching", "desc": "Aldehydes (Strong)"},
    {"min": 1725, "max": 1705, "group": "C=O stretching", "desc": "Ketones (Strong)"},
    {"min": 1700, "max": 1630, "group": "C=O stretching", "desc": "Amides (Strong)"},
    {"min": 1620, "max": 1680, "group": "C=C stretching", "desc": "Alkene (Variable)"},
    {"min": 1450, "max": 1600, "group": "C=C stretching", "desc": "Aromatic Ring (Variable)"},
    
    # Fingerprint / Single Bond Region
    {"min": 1350, "max": 1480, "group": "C-H bending", "desc": "Alkane (Variable)"},
    {"min": 1163, "max": 1210, "group": "C-O stretching", "desc": "Ester (Strong, Sharp)"},
    {"min": 1124, "max": 1205, "group": "C-O stretching", "desc": "Tertiary Alcohol (Strong, Sharp)"},
    {"min": 1087, "max": 1124, "group": "C-O stretching", "desc": "Secondary Alcohol (Strong, Sharp)"},
    {"min": 1085, "max": 1150, "group": "C-O stretching", "desc": "Aliphatic Ether (Strong, Sharp)"},
    {"min": 1050, "max": 1085, "group": "C-O stretching", "desc": "Primary Alcohol (Strong, Sharp)"},
    {"min": 1040, "max": 1050, "group": "CO-O-CO stretching", "desc": "Anhydride (Strong, Broad)"},
    {"min": 1030, "max": 1070, "group": "S=O stretching", "desc": "Sulfoxide (Strong, Sharp)"},
    
    # Bending / Halo Region
    {"min": 985,  "max": 995,  "group": "C=C bending", "desc": "Allene (Strong, Sharp)"},
    {"min": 970,  "max": 990,  "group": "=C–H oop bend", "desc": "Trans-alkene (Strong, Sharp)"},
    {"min": 960,  "max": 980,  "group": "C=C bending", "desc": "Alkene (Strong, Sharp)"},
    {"min": 905,  "max": 920,  "group": "=C–H oop bend", "desc": "Vinyl group (Medium)"},
    {"min": 650,  "max": 900,  "group": "C-H oop bend", "desc": "Aromatic Substitution (Medium-Strong)"},
    {"min": 515,  "max": 690,  "group": "C-Br stretching", "desc": "Halo Compound (Strong)"},
    {"min": 500,  "max": 600,  "group": "C-I stretching", "desc": "Halo Compound (Strong)"}
]

def identify_peak(wavenumber):
    """Checks which functional group a specific wavenumber belongs to."""
    matches = []
    for item in FUNCTIONAL_GROUPS:
        if item["min"] <= wavenumber <= item["max"]:
            matches.append(f"{item['group']} ({item['desc']})")
    
    return ", ".join(matches) if matches else "Unknown / Fingerprint Region"

# --- 2. Sidebar: Controls ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx) or CSV", type=["xlsx", "xls", "csv"])

st.sidebar.header("2. Processing Settings")
auto_flip = st.sidebar.checkbox("Auto-flip graph (Ensure peaks point up)", value=True, help="Automatically detects if graph is Transmittance (upside down) and flips it to Absorbance style.")

if not auto_flip:
    spectrum_type_manual = st.sidebar.radio("Manual Spectrum Type", ["Transmittance (%T)", "Absorbance"])
else:
    spectrum_type_manual = None

prominence = st.sidebar.slider("Peak Sensitivity (Prominence)", 0.01, 0.5, 0.05, 0.01)
distance = st.sidebar.slider("Min Distance between peaks", 1, 100, 20)

# --- 3. Main Logic ---
if uploaded_file is not None:
    try:
        # Load Data
        if uploaded_file.name.endswith('csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Pre-processing: Clean column names
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Try to find relevant columns automatically
        wavenumber_col = next((c for c in df.columns if 'wave' in c or 'cm' in c), df.columns[0])
        intensity_col = next((c for c in df.columns if 'trans' in c or 'abs' in c or 'int' in c), df.columns[1])

        # Sort data by wavenumber (standard is usually high to low, but we sort ascending for plotting logic)
        df = df.sort_values(by=wavenumber_col).reset_index(drop=True)
        
        x = df[wavenumber_col].values
        y = df[intensity_col].values

        # --- Normalization ---
        # Min-Max Normalization: Scales data between 0 and 1
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

        # --- Orientation Logic ---
        # Determine if we need to flip the signal so peaks point UP (Absorbance style)
        flip_signal = False
        y_plot = y_norm
        
        if auto_flip:
            # Heuristic: If the median is high (> 0.5), it's likely Transmittance (baseline at top)
            if np.median(y_norm) > 0.5:
                st.info("💡 Detected Transmittance signal (peaks point down). Automatically flipped to Absorbance-style (peaks point up) for analysis.")
                y_plot = 1 - y_norm # Flip signal
                flip_signal = True
            else:
                st.info("💡 Detected Absorbance signal (peaks point up). No flipping needed.")
        else:
            # Manual Mode
            if spectrum_type_manual == "Transmittance (%T)":
                # In manual Transmittance, we usually keep the plot as is (valleys), but find peaks on inverted
                pass 
            else:
                pass

        # --- Peak Picking ---
        # We need a signal where peaks are MAXIMA for find_peaks
        if auto_flip:
            # Since we already ensured y_plot has peaks pointing UP
            peaks, properties = find_peaks(y_plot, prominence=prominence, distance=distance)
            plot_y = y_plot # We plot the potentially flipped version
            ylabel = "Normalized Intensity (Auto-Corrected)"
        else:
            # Manual handling
            if spectrum_type_manual == "Transmittance (%T)":
                # Peaks are valleys (minima), so we invert for search
                peaks, properties = find_peaks(-y_norm, prominence=prominence, distance=distance)
                plot_y = y_norm # Plot original
                ylabel = "Transmittance (Normalized)"
            else:
                peaks, properties = find_peaks(y_norm, prominence=prominence, distance=distance)
                plot_y = y_norm
                ylabel = "Absorbance (Normalized)"

        # --- Visualization ---
        st.subheader("Normalized FTIR Spectrum")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, plot_y, label='Signal', color='#2980b9', linewidth=1.5)
        
        # Mark Peaks
        ax.plot(x[peaks], plot_y[peaks], "x", color='#e74c3c', label='Identified Peaks')
        
        # Annotate Peaks
        for p in peaks:
            # Decide label position
            if not auto_flip and spectrum_type_manual == "Transmittance (%T)":
                # Text below for valleys
                xytext = (0, -15)
                va = 'top'
            else:
                # Text above for hills
                xytext = (0, 10)
                va = 'bottom'

            ax.annotate(f"{x[p]:.0f}", 
                        xy=(x[p], plot_y[p]), 
                        xytext=xytext, 
                        textcoords='offset points', 
                        rotation=90, 
                        ha='center', 
                        va=va,
                        fontsize=8, 
                        color='#c0392b')

        ax.set_xlabel("Wavenumber ($cm^{-1}$)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(max(x), min(x)) # Reverse X-axis
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

        st.pyplot(fig)

        # --- Analysis Table ---
        st.subheader("Peak Analysis & Functional Group Assignment")
        
        results = []
        for p_index in peaks:
            wn = x[p_index]
            intensity = plot_y[p_index]
            assignment = identify_peak(wn)
            
            results.append({
                "Wavenumber (cm⁻¹)": round(wn, 2),
                "Intensity": round(intensity, 4),
                "Potential Assignment": assignment
            })
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Analysis Report", csv, "ftir_analysis.csv", "text/csv", key='download-csv')
        else:
            st.warning("No peaks found. Try adjusting sensitivity.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please ensure your Excel file has two columns: Wavenumber and Intensity.")

else:
    st.info("Please upload an FTIR data file (Excel or CSV) on the left sidebar to begin.")
    
    if st.button("Generate Sample Data"):
        x_dummy = np.linspace(400, 4000, 1000)
        y_dummy = np.ones_like(x_dummy) * 0.9 
        
        def add_peak(x, center, depth, width):
            return -depth * np.exp(-(x - center)**2 / (2 * width**2))

        y_signal = (
            add_peak(x_dummy, 3400, 0.4, 50) +  
            add_peak(x_dummy, 2950, 0.3, 30) +  
            add_peak(x_dummy, 1710, 0.5, 15) +  
            add_peak(x_dummy, 1450, 0.2, 10)    
        )
        
        noise = np.random.normal(0, 0.005, len(x_dummy))
        y_final = y_dummy + y_signal + noise
        
        dummy_df = pd.DataFrame({"Wavenumber": x_dummy, "Transmittance": y_final})
        dummy_df.to_excel("sample_ftir.xlsx", index=False)
        st.success("Generated 'sample_ftir.xlsx'. You can download it and upload it to the sidebar to test!")
        
        with open("sample_ftir.xlsx", "rb") as f:
            st.download_button("Download Sample Excel", f, "sample_ftir.xlsx")