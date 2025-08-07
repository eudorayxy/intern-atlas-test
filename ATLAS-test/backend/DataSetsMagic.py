Dids_dict = {
    # Z->ll
    'Zee' : [700320, 700321, 700322],
    'Zmumu' : [700323, 700324, 700325],
    'Ztautau' : [700792, 700793, 700794],
    # VBF Z->ll
    'VBF_Zee' : [700358], 'VBF_Zmumu' : [700359],
    'VBF_Ztautau' : [700360],
    # W->lv
    'Wenu' : [700338, 700339, 700340],
    'Wmunu' : [700341, 700342, 700343],
    'Wtaunu' : [700344, 700345, 700346, 700347, 700348, 700349],
    # VBF W->lv
    'VBF_Wenu' : [700362],
    'VBF_Wmunu' : [700363],
    'VBF_Wtaunu' : [700364],
    # ttbar
    'ttbar' : [410470],
    # VV->llll
    'VV4l' : [700600, 700587, 700591],
    # Low-mass Drell-Yan samples
    'm10_40_Zee' : [700467, 700468, 700469],
    'm10_40_Zmumu' : [700470, 700471, 700472],
    # H->yy
    'ggF_Hyy' : [343981], 'VBF_Hyy' : [346214], 'WpH_Hyy' : [345318],
    'WmH_Hyy' : [345317], 'ZH_Hyy' : [345319], 'ggZH_Hyy' : [345061],
    'ttH_Hyy' : [346525],
    # H->llll
    'ggH_H4l' : [345060], 'VBF_H4l' : [346228],
    'ttH_H4l' : [346340, 346341, 346342],
    'ggZH_H4l' : [345066], 'ZH_H4l' : [346645],
    'WpH_H4l' : [346646], 'WmH_H4l' : [346647],
    }

# Create records with generic name
Dids_dict['VBF_Zll'] = (Dids_dict['VBF_Zee'] + 
                        Dids_dict['VBF_Zmumu'] + 
                        Dids_dict['VBF_Ztautau'])
Dids_dict['m10_40_Zll'] = (Dids_dict['m10_40_Zee'] + 
                           Dids_dict['m10_40_Zmumu'])
Dids_dict['Wlepnu'] = (Dids_dict['Wenu'] + 
                       Dids_dict['Wmunu'] + 
                       Dids_dict['Wtaunu'])
Dids_dict['VBF_Wlepnu'] = (Dids_dict['VBF_Wenu'] + 
                           Dids_dict['VBF_Wmunu'] + 
                           Dids_dict['VBF_Wtaunu'])
Dids_dict['Hyy'] = (Dids_dict['ggF_Hyy'] + 
                    Dids_dict['VBF_Hyy'] +
                    Dids_dict['WpH_Hyy'] +
                    Dids_dict['WmH_Hyy'] + 
                    Dids_dict['ZH_Hyy'] +
                    Dids_dict['ggZH_Hyy'] +
                    Dids_dict['ttH_Hyy'])
Dids_dict['H4l'] = (Dids_dict['ggH_H4l'] + 
                      Dids_dict['VBF_H4l'] +
                      Dids_dict['WpH_H4l'] +
                      Dids_dict['WmH_H4l'] + 
                      Dids_dict['ZH_H4l'] +
                      Dids_dict['ggZH_H4l'] +
                      Dids_dict['ttH_H4l'])
                    
validSkims = ['2to4lep', '2muons', 'GamGam', 'exactly4lep', '1LMET30', '3J1LMET30', 
              '2J2LMET30', '2bjets', '3lep', 'exactly3lep', '4lep']
