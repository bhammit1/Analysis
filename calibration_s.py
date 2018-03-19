import os
from tools.functions_idm import idm_sensitivity_analysis_file
from tools.functions_gipps import gipps_sensitivity_analysis_file
from tools.functions_w99 import w99_sensitivity_analysis_file
from tools.functions_newell import newell_sensitivity_analysis_file

# Identify save paths and save name scheme:
# ----------------------------------------------------------------------------------------
save_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\04-Dissertation\Rachel_Britton\03-Projects\2018.02 Model Calibration\IDM\Sensitivity Analysis_2'
date = '2018-02-09_idm_s2'  # Naming for output files - Cannot be with dots - must be with dashes

# GA Inputs - Identified from sensitivity analysis!
# ----------------------------------------------------------------------------------------
CXPB, MUTPB, M_INDPB, NGEN, NPOP, NRUNS = [0.6], [0.4], [0.4], [60,80,100], [600,800,100], 5

summary_file = open(os.path.join(save_path,'{}_summary.csv'.format(date)),'r')

idm_sensitivity_analysis_file(summary_file, date, save_path, CXPB, MUTPB, M_INDPB, NGEN, NPOP)