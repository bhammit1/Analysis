import os
from tools.functions_gipps import gipps_convergence_plot
from tools.functions_gipps import gipps_sensitivity_analysis_plot
from tools.functions_gipps import gipps_sensitivity_analysis_file
from tools.functions_w99 import w99_convergence_plot
from tools.functions_w99 import w99_sensitivity_analysis_plot
from tools.functions_w99 import w99_sensitivity_analysis_file
from tools.functions_idm import idm_convergence_plot
from tools.functions_idm import idm_sensitivity_analysis_plot
from tools.functions_idm import idm_sensitivity_analysis_file

# Inputs
# ----------------------------------------------------------------------------------------
model = 'idm'
input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\04-Dissertation\Rachel_Britton\03-Projects\2018.02 Model Calibration\IDM\Sensitivity Analysis_3'
save_path = input_path
date = '2018-02-12_idm_s3'

# GA Inputs
# ----------------------------------------------------------------------------------------
CXPB, MUTPB, M_INDPB, NGEN, NPOP, NRUNS = [0.7], [0.3], [0.5], [60,80,100], [600,800,1000], 5

# Runs to consider for convergence analysis
# ----------------------------------------------------------------------------------------
runs = []

# Open Calibration Summary File (Sensitivity Analysis)
# ----------------------------------------------------------------------------------------
summary_filename = '{}_summary.csv'.format(date)
summary_file = open(os.path.join(input_path,summary_filename),'r')

# Generate Plots/Summary Sheets:
# ----------------------------------------------------------------------------------------
if model == 'gipps':
    gipps_sensitivity_analysis_plot(summary_file=summary_file,date=date,save_path=save_path,CXPB=CXPB,MUTPB=MUTPB,NGEN=NGEN,NPOP=NPOP)

    gipps_sensitivity_analysis_file(summary_file=summary_file, date=date, save_path=save_path, CXPB=CXPB,
                                         MUTPB=MUTPB, M_INDPB=M_INDPB, NGEN=NGEN, NPOP=NPOP)

    for run in runs:
        log_filename = '{}_{}_logfile.csv'.format(date, run)
        log_file = open(os.path.join(input_path, log_filename), 'r')
        gipps_convergence_plot(log_file=log_file, date=date + '_{}'.format(run), save_path=save_path)

elif model == 'w99':
    w99_sensitivity_analysis_plot(summary_file=summary_file, date=date, save_path=save_path, CXPB=CXPB, MUTPB=MUTPB,
                                    NGEN=NGEN, NPOP=NPOP)

    w99_sensitivity_analysis_file(summary_file=summary_file, date=date, save_path=save_path, CXPB=CXPB,
                                    MUTPB=MUTPB, M_INDPB=M_INDPB, NGEN=NGEN, NPOP=NPOP)

    for run in runs:
        log_filename = '{}_{}_logfile.csv'.format(date, run)
        log_file = open(os.path.join(input_path, log_filename), 'r')
        w99_convergence_plot(log_file=log_file, date=date + '_{}'.format(run), save_path=save_path)

elif model == 'idm':


    idm_sensitivity_analysis_file(summary_file=summary_file, date=date, save_path=save_path, CXPB=CXPB,
                                    MUTPB=MUTPB, M_INDPB=M_INDPB, NGEN=NGEN, NPOP=NPOP)





