from mmengine.registry import Registry


MODELS = Registry('model', locations=['modules'])
ADAPTERS = Registry('adapter', locations=['modules.adapter'])
DATASETS = Registry('dataset', locations=['dataloaders'])
RUNNERS = Registry('runner', locations=['runner'])
