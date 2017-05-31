import pickle

import numpy as np

DEFAULT_FILENAME = "tensor_shapes.txt"
FILENAME_WITH_STEP = "tensor_shapes-%d.txt"


class GraphMeta:
    def __init__(self, starting_checkpoint):
        self.step = self.get_step_from_starting_checkpoint(starting_checkpoint)
        self.filename = FILENAME_WITH_STEP % self.step
        self.dict = dict()
        if self.filename is not None:
            try:
                with open(self.filename, 'rb') as handle:
                    self.dict = pickle.loads(handle.read())
            except IOError:
                pass

    def get_variable_shape(self, name):
        if not name.endswith(":0"):
            name = "%s:0" % name
        return self.as_list(self.dict.get(name, None))

    def set_variable_shape(self, name, shape):
        if not name.endswith(":0"):
            name = "%s:0" % name
        self.dict.update({name: self.as_list(shape)})

    def save(self, new_step):
        new_file = FILENAME_WITH_STEP % new_step
        with open(new_file, 'wb') as handle:
            pickle.dump(self.dict, handle)

    @staticmethod
    def as_list(shape):
        shape_type = type(shape)
        if shape_type is list:
            return shape
        elif shape_type is np.ndarray:
            return shape.tolist()

    @staticmethod
    def get_step_from_starting_checkpoint(starting_checkpoint):
        if starting_checkpoint is not None and '-' in starting_checkpoint:
            return int(starting_checkpoint.split('-')[-1])
        else:
            return 0
