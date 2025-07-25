Dids_2to4lep = {
    'Zee' : [700320,700321,700322],
    'Zmumu' : [700323, 700324, 700325],
    'Ztautau' : [700792, 700793, 700794],
    'Wlepnu' : [700338, 700339, 700340, # W->enu
                700341, 700342, 700343, # W->munu
                700344, 700345, 700346, 700347, 700348, 700349], # W->taunu
    'ttbar' : [410470, # ttbar
               411234, 601491], # ttbar->2lep
    'H' : [345060, 346228, 346340, 346341, 346342], # H->ZZ->llll
    'ZZllll' : [700600, 700601] # ZZ*
    }

Dids_Hyy = {
    'ggF' : 343981,
    'VBF' : 346214,
    'WpH' : 345318,
    'WmH' : 345317,
    'ZH' : 345319,
    'ggZH' : 345061,
    'ttH' : 346525
}
validSkims = ['2to4lep', '2muons', 'GamGam', 'exactly4lep', '1LMET30', '3J1LMET30', 
              '2J2LMET30', '2bjets', '3lep', 'exactly3lep', '4lep']

DIDS_ALL = {
    '2to4lep': Dids_2to4lep
}