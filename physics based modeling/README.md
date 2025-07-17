# Simulink
Siculink and simscape code for generating data. 

1. run_experiments_v2.m
    * Code runs and creates 100 CSV files in one of the following folders
        * with_friction
        * no_friction
    * Change "with_friction = " on line 13 in the code to create either or. 
1. run_pinn_v1.m 
    * Run to create PINN dataset.
    * as is, it only creates 1 csv file, I need 100, correct? 
    * Warning: Source 'pinn_degrade_v3/From Workspace1' specifies that its sample time (-1) is back-inherited. You should explicitly specify the sample time of sources. You can disable this diagnostic by setting the 'Source block specifies -1 sample time' diagnostic to 'none' in the Sample Time group on the Diagnostics pane of the Configuration Parameters dialog box. 
1. to_numpy.py
    * Run to convert .csv files to .npy.
    * This code creates three files in the "PIML examples\data" data folder, pinn_test_0.npy, with_friction.npz and no_friction.npz.
