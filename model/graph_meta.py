import pickle

DEFAULT_FILENAME = "variable_shapes.txt"
FILENAME_WITH_STEP = "variable_shapes-%d.txt"


class GraphMeta:
    def __init__(self, starting_checkpoint):
        self.step = self.get_step_from_starting_checkpoint(starting_checkpoint)
        self.filename = starting_checkpoint
        self.dict = dict()
        if self.filename is not None:
            with open(self.filename, 'rb') as handle:
                self.dict = pickle.loads(handle.read())

    def get_variable_shape(self, name):
        variable_name = "%s:0" % name
        return self.dict.get(variable_name, None)

    def set_variable_shape(self, name, shape):
        variable_name = "%s:0" % name
        self.dict.update({variable_name: shape})

    def save(self, new_step):
        new_file = FILENAME_WITH_STEP % new_step
        with open(new_file, 'wb') as handle:
            pickle.dump(self.dict, handle)

    @staticmethod
    def get_step_from_starting_checkpoint(starting_checkpoint):
        if starting_checkpoint is not None and starting_checkpoint.contains('-'):
            return int(starting_checkpoint.split('-')[-1])
        else:
            return 0
