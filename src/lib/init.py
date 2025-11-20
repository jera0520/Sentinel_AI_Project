

DETECT_MODEL = 'person5l'

model_cfgs = {
    'person5l':{
        'names': 'model/person5l/model.names',
        'cfg': 'model/person5l/model.cfg',
        'weights': 'model/person5l/model.weights'
    },
    'helmet_resort_v2':{
        'names': 'model/helmet_resort_v2/model.names',
        'cfg': 'model/helmet_resort_v2/model.cfg',
        'weights': 'model/helmet_resort_v2/model.weights'
    },
    'falldown_v3':{
        'names': 'model/falldown_v3/model.names',
        'cfg': 'model/falldown_v3/model.cfg',
        'weights': 'model/falldown_v3/model.weights'
    },
}

color_cfgs = {
    'person5l': (255, 0, 0),
    'wearing_helmet': (0, 255, 0),
    'detecting_helmet': (255, 100, 255),
    'nohelmet': (0, 0, 255),
}