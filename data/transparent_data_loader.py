class TransparentDataLoader:
    """
    This class represents both dataloader and a training session that can be saved and loaded
    """
    def __init__(self):
        super(TransparentDataLoader, self).__init__()

    def init_epoch(self, epoch, batch_size):
        raise NotImplementedError

    def get_next_batch(self):
        raise NotImplementedError

    def set_epoch_it(self, epoch):
        raise NotImplementedError

    def get_epoch_it(self):
        raise NotImplementedError

    def get_num_epoch(self):
        raise NotImplementedError

    def get_num_batches(self):
        raise NotImplementedError

    def set_batch_it(self, batch_it):
        raise NotImplementedError

    def get_batch_it(self):
        raise NotImplementedError

    def get_batch_size(self):
        raise NotImplementedError

    def save_state(self):
        raise NotImplementedError

    def load_state(self, state_dict):
        raise NotImplementedError
