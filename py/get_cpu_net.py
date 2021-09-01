from net import *
import sys

net = NeuralNetWorkWrapper(lr=0.001, l2=0.0001, num_layers=8, num_channels=64, n=8, action_size=65)
if len(sys.argv) == 3:
    net.load_model("../models", str(sys.argv[1]))
    net.libtorch_use_gpu = False
    net.save_model("../models", "jws_cpu_ver_" + str(sys.argv[2]))