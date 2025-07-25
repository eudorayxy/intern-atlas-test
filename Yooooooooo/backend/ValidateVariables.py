# Reference for input validation (Needs updates for relevant physics processes)
tree_variables_ref = ['trigE', 'trigM', 'lep_n', 'lep_pt', 'lep_eta', 'lep_phi', 'lep_e', 'lep_charge', 'lep_type', 'lep_ptvarcone30', 'lep_topoetcone20', 'lep_isTrigMatched', 'lep_isLooseID', 'lep_isMediumID', 'lep_isTightID', 'lep_isLooseIso', 'lep_isTightIso', 'photon_n', 'photon_pt', 'photon_eta', 'photon_phi', 'photon_e', 'photon_ptcone20', 'photon_topoetcone40', 'photon_isLooseID', 'photon_isTightID', 'photon_isLooseIso', 'photon_isTightIso', 'met', 'met_phi', 'jet_n', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_e', 'jet_btag_quantile']

def validate_variables(variables_list_input):
    validated_variables_list = []
    for variable_input in variables_list_input:
        if variable_input not in tree_variables_ref:
            print(f"Skipping '{variable_input}' - invalid input")
        elif variable_input in validated_variables_list:
            print(f"Skipping '{variable_input}' - duplicated entry")
        else:
            validated_variables_list.append(variable_input)
    return validated_variables_list