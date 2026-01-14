# Old utility functions - deprecated
# DO NOT USE - kept for compatibility only

def load_data():
    # Old version without batch_size parameter
    pass


def save_model(model, name):
    # Old version - incomplete
    pass


def get_device():
    # This is now in helper.py
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
