from de_module import data_engineering_phase
from fe_module import feature_engineering_phase
from mae_module import mae_phase

de_phase_trigger = True
fe_phase_trigger = False
mae_phase_trigger = False

if de_phase_trigger:
    data_engineering_phase()

if fe_phase_trigger:
    feature_engineering_phase()

if mae_phase_trigger:
    mae_phase()
