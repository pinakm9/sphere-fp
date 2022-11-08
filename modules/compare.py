from xml import dom
import tensorflow as tf 
import matplotlib.pyplot as plt
import arch
import csv

def plot_difference_2D(net, steps, truth, domain, net_path, num_pts=10000, num_mc_pts=int(1e7), **net_kwargs):
    pts = tf.random.uniform(low=domain[0], high=domain[1], size=(num_pts, len(domain[0])))
    mc_pts = tf.random.uniform(low=domain[0], high=domain[1], size=(num_mc_pts, len(domain[0])))
    x, y = tf.split(pts, 2, axis=-1)
    mc_x, mc_y = tf.split(mc_pts, 2, axis=-1)
    net = arch.LSTMForgetNet(**net_kwargs)
    true = truth(x, y) 
    with open('{}/difference.csv'.format(net_path), 'w') as logger:
        writer = csv.writer(logger)
        for step in steps:
            net.load_weights("{}/{}_{}".format(net_path, net.name, step))
            Z_mc = (domain[1][0] - domain[0][0]) * (domain[1][1] - domain[0][1]) \
                   * tf.reduce_mean(tf.exp(net(mc_x, mc_y))).numpy()
    `       learned = tf.exp(net(x, y)) / Z_mc
            difference = tf.max(tf.abs(true - learned)).numpy()
            writer.writerow(difference)









