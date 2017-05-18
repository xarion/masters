import itertools


class flags:
    def __init__(self, arglist):
        self.arglist = arglist
        self.debugging = arglist[0]
        self.prune_on_activation_count = arglist[1]
        self.distort_weights = arglist[2][0]
        self.distort_by_kernel = arglist[2][1]
        self.prune_outliers = arglist[3][0]
        self.std_multiplier = arglist[3][1]
        self.activation_threshold = arglist[3][1]
        self.l1_regularization = arglist[4][0]
        self.l1_regularization_multiplier = arglist[4][1]
        self.activation_count_regularization = arglist[5][0]
        self.activation_count_multiplier = arglist[5][1]
        self.increment_epochs_after_traning_cycle = arglist[6]
        self.plot_figures = arglist[7]
        self.initial_epochs = arglist[8]
        self.seed = arglist[9]

    def __str__(self):
        return """CUDA_VISIBLE_DEVICES=1 python activation_based_pruning_cnn_mnist.py --debugging %s \\
        --prune_on_activation_count %s \\
        --distort_weights %s \\
        --distort_by_kernel %s \\
        --prune_outliers %s \\
        --std_multiplier %s \\
        --activation_threshold %s \\
        --l1_regularization %s \\
        --l1_regularization_multiplier %s \\
        --activation_count_regularization %s \\
        --activation_count_multiplier %s \\
        --increment_epochs_after_traning_cycle %s \\
        --plot_figures %s \\
        --initial_epochs %s \\
        --seed %s >> experiments
        """ % (self.arglist[0], self.arglist[1], self.arglist[2][0], self.arglist[2][1], self.arglist[3][0],
               self.arglist[3][1], self.arglist[3][1], self.arglist[4][0], self.arglist[4][1], self.arglist[5][0],
               self.arglist[5][1], self.arglist[6], self.arglist[7], self.arglist[8], self.arglist[9])


debug = [False]
prune_on_activation_count = [True, False]
distort_weights_and_kernel = [(True, True), (True, False), (False, False)]
prune_outliers_and_std_multiplier_or_threshold = [(True, 2), (True, 4), (False, 0), (False, 3600), (False, 12000)]
l1_regularization = [(True, 0.00001), (True, 0.000001), (False, 0)]
activation_count_regularization = [(True, 0.00001), (True, 0.000001), (False, 0)]
increment_epochs_after_traning_cycle = [True, False]
plot_figures = [False]
initial_epochs = [2]
seed = [1234, 127, 421]

iterator = itertools.product(debug, prune_on_activation_count, distort_weights_and_kernel,
                             prune_outliers_and_std_multiplier_or_threshold, l1_regularization,
                             activation_count_regularization, increment_epochs_after_traning_cycle,
                             plot_figures, initial_epochs, seed)

total = len(debug) * len(prune_on_activation_count) * len(distort_weights_and_kernel) * \
        len(prune_outliers_and_std_multiplier_or_threshold) * len(l1_regularization) * \
        len(activation_count_regularization) * len(increment_epochs_after_traning_cycle) * \
        len(plot_figures) * len(initial_epochs) * len(seed)
count = 1
for i in iterator:
    f = flags(i)
    print f
    print "echo %d/%d" % (count, total)
    count += 1
