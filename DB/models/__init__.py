from . import model, loss


def get_model(config):
    _model = getattr(model, config['type'])(config['args'])
    return _model


def get_loss(config):
    return getattr(loss, config['type'])(**config['args'])
