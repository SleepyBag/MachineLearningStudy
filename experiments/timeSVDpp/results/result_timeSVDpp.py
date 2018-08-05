n [1]: run main.py
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
1
bin_cnt (default 30):
beta (default 0.4):
factor_cnt (default 10):
batch_size (default 40):
learning_method:(default sgd)adam
beta1 (default 0.9):
beta2 (default 0.999):
wd (default 0.01):
epoch_cnt (default 20):3
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.9818908138446454
Test finished, Loss = 0.9532935301823358
Test finished, Loss = 0.9431807057770001
continue?(y/n)y
learning_method:(default sgd)
learning_rate (default 0.001):
wd (default 0.01):
epoch_cnt (default 20):
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.9429706529599688
Test finished, Loss = 0.9428028152469105
Test finished, Loss = 0.9426691865720975
Test finished, Loss = 0.9425501996743992
Test finished, Loss = 0.9424512483246813
Test finished, Loss = 0.9423686600297413
Test finished, Loss = 0.9422848726632514
Test finished, Loss = 0.9422153810136777
Test finished, Loss = 0.9421497193917748
Test finished, Loss = 0.9420904282498576
Test finished, Loss = 0.94203346584994
Test finished, Loss = 0.9419789881215503
Test finished, Loss = 0.9419413028255794
Test finished, Loss = 0.9418860939957124
Test finished, Loss = 0.9418486641971734
Test finished, Loss = 0.9418099886322613
Test finished, Loss = 0.94177333351436
Test finished, Loss = 0.9417375065537106
Test finished, Loss = 0.9417019893356656
Test finished, Loss = 0.9416696337987914
continue?(y/n)y
learning_method:(default sgd)
learning_rate (default 0.001):
wd (default 0.01):
epoch_cnt (default 20):40
verbose (default 15):
progressbar? (default 0):^C---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
~/MachineLearningStudy/experiments/timeSVDpp/main.py in <module>()
127
128 trainer = get_trainer()
--> 129 train(trainer)

~/MachineLearningStudy/experiments/timeSVDpp/main.py in train(trainer)
119             epoch_cnt = input_param('epoch_cnt', 20)
120             verbose = input_param('verbose', 15)
--> 121             progress = bool(input_param('progressbar?', 0))
122             trainer.train(epoch_cnt, learning_method, learning_params, verbose,
123                           progress=progress)

~/MachineLearningStudy/experiments/timeSVDpp/main.py in input_param(prompt, default)
46
47 def input_param(prompt, default):
---> 48     ans = input(prompt + (' (default ' + str(default) + '):'))
49     if ans != '':
50         ans = float(ans)

KeyboardInterrupt:

In [2]: train(trainer)
learning_method:(default sgd)
learning_rate (default 0.001):
wd (default 0.01):
epoch_cnt (default 20):
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.941636136350581
Test finished, Loss = 0.9416032599873208
Test finished, Loss = 0.9415752052743849
Test finished, Loss = 0.9415415488873818
Test finished, Loss = 0.9415129218610067
Test finished, Loss = 0.9414851237705512
Test finished, Loss = 0.9414631855189559
Test finished, Loss = 0.9414323258836771
Test finished, Loss = 0.9414096083964992
Test finished, Loss = 0.9413867866245256
Test finished, Loss = 0.9413686844164724
Test finished, Loss = 0.9413485070465402
Test finished, Loss = 0.9413246463594533
Test finished, Loss = 0.941296220229716
Test finished, Loss = 0.9412755225623367
Test finished, Loss = 0.941254668813122
Test finished, Loss = 0.9412369272000541
Test finished, Loss = 0.9412146200189944
Test finished, Loss = 0.9411896145968675
Test finished, Loss = 0.9411729612397235
continue?(y/n)y
learning_method:(default sgd)
learning_rate (default 0.001):.005
wd (default 0.01):
epoch_cnt (default 20):40
verbose (default 15):0
progressbar? (default 0):
Test finished, Loss = 0.9410945671796219
Test finished, Loss = 0.9410324595059142
Test finished, Loss = 0.9409357356343472
Test finished, Loss = 0.9409178323251718
Test finished, Loss = 0.94094850126415
Test finished, Loss = 0.9408672341748064
Test finished, Loss = 0.9408671822778707
Test finished, Loss = 0.9407203543860684
Test finished, Loss = 0.9406907161429309
Test finished, Loss = 0.9407526906684881
Test finished, Loss = 0.940703329379805
Test finished, Loss = 0.9406835010877729
Test finished, Loss = 0.9406659044262208
Test finished, Loss = 0.9406076098890812
Test finished, Loss = 0.9407401299973867
Test finished, Loss = 0.940667254034337
Test finished, Loss = 0.9408108205160057
Test finished, Loss = 0.9407284515370257
Test finished, Loss = 0.9407539363457376
