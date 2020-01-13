import os
import json

config_path = 'resources/cfg/config.json'


plate_score_thresshold = 0.5
plate_iou_thresshold = 0.0


class Config:
    __instance = None

    def __init__(self, mode='local'):
        """ Virtually private constructor. """
        if Config.__instance is not None:
            Config.getInstance()
        else:
            self.mode = mode
            self.load_cfg_file()
            Config.__instance = self

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Config.__instance is None:
            Config()
        return Config.__instance

    def load_cfg_file(self):
        with open(config_path) as json_file:
            data = json.load(json_file)
            if self.mode == 'local':
                data = data['local']
            elif self.mode == 'prod':
                data = data['prod']
            elif self.mode == 'training':
                data = data['training']
            self.plateImages = data['plateImages']
            self.plateAnnotations = data['plateAnnotations']
            self.plateLabels = data['plateLabels']
            self.yoloWeights = data['yoloWeights']
            self.yoloAnchors = data['yoloAnchors']
            self.yoloTopless = data['yoloTopless']
            self.yoloH5Weights = data['yoloH5Weights']
            self.plateDetectWeight = data['plateDetectWeight']

if __name__ == '__main__':
    Config('local')
    config = Config.getInstance()
    # Config('prod')