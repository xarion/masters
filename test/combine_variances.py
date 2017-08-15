import tensorflow as tf

batches = 4
batch_size = 2
s = tf.Session()
a = tf.truncated_normal([batches, 24, 24, 32])

previous_means = tf.zeros([32])
previous_stds = tf.zeros([32])
previous_counts = tf.zeros([32])
real_mean, real_std = tf.nn.moments(a, axes=[0, 1, 2])

for x in range(0, batches, batch_size):
    mini_batch = a[x:(batch_size + x)]
    current_means, current_stds = tf.nn.moments(mini_batch, axes=[0, 1, 2])
    new_counts = previous_counts + 1

    new_means = (previous_means * previous_counts + current_means) / new_counts

    new_stds = ((((previous_stds) + tf.square(previous_means - new_means)) * previous_counts
                 + ((current_stds) + tf.square(current_means - new_means)))
                / new_counts)

    previous_means = new_means
    previous_stds = new_stds
    previous_counts = new_counts

s.run(tf.variables_initializer(tf.global_variables()))

diff_mean, diff_std = s.run([real_mean - previous_means, real_std - previous_stds])

print diff_mean
print diff_std
