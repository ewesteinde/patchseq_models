# be in allesdk environment
from allensdk.api.queries.biophysical_api import BiophysicalApi

# download neuron model
bp = BiophysicalApi()
bp.cache_stimulus = True # change to False to not download the large stimulus NWB file
neuronal_model_id = 473862496    # L2/3 pyr neuron in VISp from Nr5a1-Cre line https://celltypes.brain-map.org/experiment/electrophysiology/382982932
bp.cache_data(neuronal_model_id, working_directory='neuronal_model')