n [2]: run main.py
b  =  0
b  =  10000
b  =  20000
b  =  30000
b  =  40000
b  =  50000
b  =  60000
b  =  70000
#pos_events =  80000
max item id =  1682
b  =  0
b  =  10000
#pos_events =  20000
max item id =  1591
Choose a device(c for cpu, number for gpu(n)):0
Which model do you want to train?
1. timeSVD++    2. v1   3. v2    4. v3 5. batchified v2
4
bin_cnt (default 30):
beta (default 0.4):
factor_cnt (default 10):
batch_size (default 40):
learning_method:(default sgd)adam
beta1 (default 0.9):
beta2 (default 0.999):
wd (default 0.01):
epoch_cnt (default 20):2
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.9565197740376307
Test finished, Loss = 0.9364610910230067
continue?(y/n)y
learning_method:(default sgd)
learning_rate (default 0.001):
wd (default 0.01):
epoch_cnt (default 20):
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.9336415390488472
Test finished, Loss = 0.9351289286944608
In [3]: train(trainer)
learning_method:(default sgd)
learning_rate (default 0.001):.0005
wd (default 0.01):
epoch_cnt (default 20):
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.9331626750599276
Test finished, Loss = 0.9338912836553835
In [4]: train(trainer)
learning_method:(default sgd)
learning_rate (default 0.001):.0001
wd (default 0.01):
epoch_cnt (default 20):
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.9339306531235336

