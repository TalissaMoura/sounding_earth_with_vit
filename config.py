from yacs.config import CfgNode as CN

cfg = CN()

cfg.SoundEncoder = 'ResNet18'
cfg.SoundReducer = 'SoundMeanPool'

cfg.ImageEncoder = 'ResNet18'
cfg.ImageReducer = 'ImageMeanPool'

cfg.LossFunction = 'TripletLoss'
cfg.LossArg      = []
cfg.LatentDim    = 128

cfg.MarginScaling = False
cfg.LocalizedSampling = False
cfg.DataRoot = './datasets/SoundingEarth/data'

cfg.MaxSamples = 1
cfg.Epochs = 100
cfg.EarlyStopping = float('inf')
cfg.BatchSize = 32
cfg.DataThreads = 2
cfg.AugmentationMode = 'image'
cfg.AudioAugmentationMode = True

cfg.Optimizer = CN()
cfg.Optimizer.Name = 'Adam'
cfg.Optimizer.LearningRate = 1e-4

cfg.RunId = ''
cfg.ModelName = ''

# Global Singleton to track training state
state = CN()
state.Epoch = 0
state.BoardIdx = 0
state.BestLoss = float('inf')
state.BestEpoch = -10000
