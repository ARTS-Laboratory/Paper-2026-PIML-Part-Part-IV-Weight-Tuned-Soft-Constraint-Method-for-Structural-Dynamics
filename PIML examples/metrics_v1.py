import numpy as np
import matplotlib.pyplot as plt
"""
Functions for metric results.
"""

""" signal to noise ratio """
def snr(sig, pred, dB=True):
    noise = sig - pred
    a_sig = np.sqrt(np.mean(np.square(sig)))
    a_noise = np.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*np.log10(snr)
""" root mean squared error """
def rmse(sig, pred, squared=False):
    error = sig - pred
    num = np.sum(np.square(error))
    denom = np.size(sig)
    e = num/denom
    if(squared):
        return e
    return np.sqrt(e)
""" root relative squared error """
def rrse(sig, pred):
    error = sig - pred
    mean = np.mean(sig)
    num = np.sum(np.square(error))
    denom = np.sum(np.square(sig-mean))
    return np.sqrt(num/denom)
""" normalized root mean squared error """
def nrmse(sig, pred):
    return rmse(sig, pred)/(np.max(sig)-np.min(sig))
""" time response assurance criterion """
def trac(sig, pred):
    num = np.square(np.sum(sig * pred))
    denom = np.sum(sig * sig) * np.sum(pred * pred)
    return num/denom
""" mean absolute error """
def mae(sig, pred):
    return np.sum(np.abs(sig-pred))/sig.size
#%%
metric_funcs= [snr, rmse, mae, rrse, nrmse, trac]

# put metric data for each model in a table
# models are: pure physics, pure nn, indirect, pinn, pinn control, 
# delta model 1, delta learning, delta learning control, informed architecture
metrics_table = np.zeros((9, len(metric_funcs)))
#%% pure physics
m = 1
c = 0.2
all_data = np.load('./data/with_friction.npz')['data'][:,:-1,:]

t = all_data[0,:,0]
x = all_data[:,:,1]
v = all_data[:,:,2]
a = all_data[:,:,3]
k = all_data[:,:,4]
F = all_data[:,:,5]
data_reconstructed = np.load('./model_predictions/pure_physics/k_pred.npy')[:,:,:-1]
k_pred = data_reconstructed[:,0,:]
a_pred = data_reconstructed[:,1,:]

metrics_table[0] = np.array([m(k, k_pred) for m in metric_funcs])

# cumulatitive RMSE error over time for all experiments
rmse_k = np.sqrt(np.sum((k - k_pred)**2, axis=0))
rmse_a = np.sqrt(np.sum((a - a_pred)**2, axis=0))
residual = np.sum(np.abs(F - m*a - c*v - k*x), axis=0)
np.save('./metric_results/pure_physics/rmse_k.npy', rmse_k)
np.save('./metric_results/pure_physics/rmse_a.npy', rmse_a)
np.save('./metric_results/pure_physics/residual.npy', residual)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(t, rmse_k)
ax1.set_ylabel('RMSE ($N/s$)')
ax2.plot(t, rmse_a)
ax2.set_ylabel(r'RMSE ($m/s^2$)')
ax3.plot(t, residual)
ax3.set_xlabel(r'time ($s$)')
ax3.set_ylabel(r'residual ($N$)')
plt.tight_layout()
#%% pure nn
all_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
a = all_data[:,:,3]
k = all_data[:,:,4]

k_test = k[80:,49:]

k_pred = np.load('./model_predictions/pure_nn/k_pred.npy')

metrics_table[1] = np.array([m(k_test, k_pred) for m in metric_funcs])
#%% indirect
all_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]

t = all_data[0,:,0]
a = all_data[:,:,3]
k = all_data[:,:,4]

k_test = k[80:,49:]

k_pred = np.load('./model_predictions/indirect/k_pred.npy')

metrics_table[2] = np.array([m(k_test, k_pred) for m in metric_funcs])
#%% pinn (and control)
test_data = np.load('./data/pinn_test_0.npy').T
# downsample by a factor of 20 so that sampling rate it 50 S/s
test_data = test_data[:,::20]
t = test_data[0,1:]
x = test_data[1,1:]
k = test_data[5,1:]
pinn_pred = np.load('./model_predictions/pinn/pred_out.npy')
x_pinn = pinn_pred[:,0]

plt.figure()
plt.plot(t, x, label='true')
plt.plot(t, x_pinn, label='pinn x')

k_pinn = pinn_pred[:,1]
k_control = np.load('./model_predictions/pinn/control_out.npy')

plt.figure()
plt.plot(t, k)
plt.plot(t, k_pinn)
plt.plot(t, k_control)

metrics_table[3] = np.array([m(k, k_pinn) for m in metric_funcs])
metrics_table[4] = np.array([m(k, k_control.flatten()) for m in metric_funcs])
#%% delta learning (and control)
with_friction_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
with_friction_data = with_friction_data[:,:-1:20,:]

k_wf = with_friction_data[:,:,4]

k_test = k_wf[80:,49:]

model_1_pred = np.load('./model_predictions/delta_learning/with_friction_model_1.npy')
delta_model_pred = np.load('./model_predictions/delta_learning/with_friction_combined_model.npy')
delta_control_pred = np.load('./model_predictions/delta_learning/with_friction_control.npy')

metrics_table[5] = np.array([m(k_test, model_1_pred) for m in metric_funcs])
metrics_table[6] = np.array([m(k_test, delta_model_pred) for m in metric_funcs])
metrics_table[7] = np.array([m(k_test, delta_control_pred) for m in metric_funcs])
#%% informed structure
all_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
all_data = all_data[:,:-1:20,:]
k = all_data[:,:,4]
k_test = k[80:]

k_pred = np.load('./model_predictions/informed_structure/k_pred.npy').squeeze()
metrics_table[8] = np.array([m(k_test, k_pred) for m in metric_funcs])
#%%
np.save('./metric_results/metrics_table', metrics_table)
# print results like latex table
# take wanted 
wanted_models = np.array([True, True, True, True, False, False, True, False, True])
wanted_metrics = np.array([True, True, True, False, False, True])
metrics_table = metrics_table[wanted_models].T[wanted_metrics].T

model_names = ['Pure physics', 'Pure NN', 'Indirect measurement', 'PINN', 'Delta learning', 'Informed architecture']
for model_name, model_n in zip(model_names, metrics_table):
    print(model_name, end=' & ')
    for i, metric_n in enumerate(model_n):
        print(str(metric_n)[:5], end='')
        if(i != 3):
            print(' & ', end='')
        else:
            print(' \\\\')
#%% for PINN and delta learning, make RMSE tables for 0-60 and 60-120s
#%% pinn
test_data = np.load('./data/pinn_test_0.npy').T
# downsample by a factor of 20 so that sampling rate it 50 S/s
test_data = test_data[:,::20]
t = test_data[0,1:]
x = test_data[1,1:]
k = test_data[5,1:]
pinn_pred = np.load('./model_predictions/pinn/pred_out.npy')
x_pinn = pinn_pred[:,0]

k_pinn = pinn_pred[:,1]
k_control = np.load('./model_predictions/pinn/control_out.npy').flatten()

k1 = k[t<60]
k2 = k[t>=60]
k_pinn1 = k_pinn[t<60]
k_pinn2 = k_pinn[t>=60]
k_control1 = k_control[t<60]
k_control2 = k_control[t>=60]

pinn_rmse1 = rmse(k1, k_pinn1)
pinn_rmse2 = rmse(k2, k_pinn2)
control_rmse1 = rmse(k1, k_control1)
control_rmse2 = rmse(k2, k_control2)

data = [[pinn_rmse1, control_rmse1], [pinn_rmse2, control_rmse2]]
labels = ['0-60 s', '60-120 s']

for i in range(2):
    print(labels[i], end = ' & ')
    for j in range(2):
        print(round(data[i][j], 2), end='')
        if(j == 0):
            print (' & ', end='')
        else:
            print('\\\\')

#%% delta learning table
with_friction_data = np.load('./data/with_friction.npz')['data']
# downsample by a factor of 20 so that sampling rate it 50 S/s
with_friction_data = with_friction_data[:,:-1:20,:]

t = with_friction_data[0,:,0]
k_wf = with_friction_data[:,:,4]
k_test = k_wf[80:,49:]

indices1 = t[49:]<60
indices2 = t[49:]>=60

k_test1 = k_test[:,indices1]
k_test2 = k_test[:,indices2]
k_tests = [k_test1, k_test2]

model_1_pred = np.load('./model_predictions/delta_learning/with_friction_model_1.npy')
delta_model_pred = np.load('./model_predictions/delta_learning/with_friction_combined_model.npy')
delta_control_pred = np.load('./model_predictions/delta_learning/with_friction_control.npy')

model_1_pred1 = model_1_pred[:,indices1]
model_1_pred2 = model_1_pred[:,indices2]
delta_model_pred1 = delta_model_pred[:,indices1]
delta_model_pred2 = delta_model_pred[:,indices2]
delta_control_pred1 = delta_control_pred[:,indices1]
delta_control_pred2 = delta_control_pred[:,indices2]

labels = ['0-60 s', '60-120 s']
k_pred_data = [[model_1_pred1, delta_model_pred1, delta_control_pred1],\
               [model_1_pred2, delta_model_pred2, delta_control_pred2]]


for i in range(2):
    print(labels[i], end = ' & ')
    for j in range(3):
        print(round(rmse(k_tests[i], k_pred_data[i][j]), 2), end='')
        if(j < 2):
            print(' & ', end='')
        else:
            print('\\\\')