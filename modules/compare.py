from xml import dom
import tensorflow as tf 
import matplotlib.pyplot as plt
import arch

def plot_difference(net, steps, truth, domain, num_pts=10000, num_mc_points=int(1e7), **net_kwargs):
    pts = tf.random.uniform(low=domain[0], high=domain[1], size=(num_pts, len(domain[0])))
    net = arch.LSTMForgetNet(**net_kwargs)
    for step in steps:
        


