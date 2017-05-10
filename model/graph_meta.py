import pickle

DEFAULT_FILENAME = "variable_shapes.txt"
FILENAME_WITH_STEP = "variable_shapes-%d.txt"


class GraphMeta:
    def __init__(self, step=None):
        self.step = step
        if step is not None:
            self.filename = FILENAME_WITH_STEP % step
        else:
            self.filename = DEFAULT_FILENAME
        try:
            with open(self.filename, 'rb') as handle:
                self.dict = pickle.loads(handle.read())
        except IOError:
            self.dict = dict()

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
