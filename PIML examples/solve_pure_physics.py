import numpy as np
from scipy.optimize import curve_fit
"""
Author: Nile Coble

Using only a physics understanding of the system when calculating k
"""




#%%
'''
Function to solve the spring-mass system. Solved with RK4.
'''
def spring_mass_system(t, m, c, k, F, x0, xdot0, dt):
    def f(xdot, x, Fi):
        xddot = -c*xdot - k*x + Fi
        return xddot, xdot
    y = np.zeros(t.size)
    xdot = xdot0
    x = x0
    for i in range(len(y)):
        xdot1, x1 = f(xdot, x, F[i])
        xdot2, x2 = f(xdot + 0.5*dt*xdot1, x + 0.5*dt*x1, F[i])
        xdot3, x3 = f(xdot + 0.5*dt*xdot2, x + 0.5*dt*x2, F[i])
        xdot4, x4 = f(xdot + dt*xdot3, x + dt*x3, F[i])
        
        xdot = xdot + dt/6*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
        x = x + dt/6*(x1 + 2*x2 + 2*x3 + x4)
        xddot = -c*xdot - k*x + F[i]
        # print(xddot.shape)
        y[i] = xddot
    return y
    
def spring_mass_system1(t, m, c, k, F, x0, xdot0, dt):
    def f(xdot, x, Fi):
        xddot = (-c*xdot - k*x + Fi)/m
        # print(xddot)
        return xddot, xdot
    y = np.zeros((t.size, 3))
    xdot = xdot0
    x = x0
    for i in range(len(y)):
        xdot1, x1 = f(xdot, x, F[i])
        xdot2, x2 = f(xdot + 0.5*dt*xdot1, x + 0.5*dt*x1, F[i])
        xdot3, x3 = f(xdot + 0.5*dt*xdot2, x + 0.5*dt*x2, F[i])
        xdot4, x4 = f(xdot + dt*xdot3, x + dt*x3, F[i])
        
        xdot = xdot + dt/6*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
        x = x + dt/6*(x1 + 2*x2 + 2*x3 + x4)
        xddot = (-c*xdot - k*x + F[i])/m
        # print(xddot)
        y[i] = [xddot, xdot, x]
    return y    
#%%
def main():
    #%%
    # load dataset
    all_data = np.load('./data/with_friction.npz')['data']
    
    t = all_data[0,:,0]
    x = all_data[:,:,1]
    v = all_data[:,:,2]
    a = all_data[:,:,3]
    k = all_data[:,:,4]
    F = all_data[:,:,5]
    
    dt = t[1] - t[0]
    N = int(1/dt) # take 1 second for a sample
    m = 1
    c = 0.2
    k_bounds = [500, 1500]
    data_pred_tot = np.zeros((k.shape[0], 2, k.shape[1]))
    
    for test in range(x.shape[0]):
        
        a_test = a[test]
        F_test = F[test]
        a_batches = a_test[1:].reshape(-1, N)
        t_batches = t[1:].reshape(-1, N)
        F_batches = F_test[1:].reshape(-1, N)
        # initial x and v. Future batches are propagated from the last solution.
        x0_batch = x[test, 0]
        v0_batch = v[test, 0]
        
        n_batches = a_batches.shape[0]
        # contains [k, a]
        data_pred = np.zeros((2, a_test.size))
        for j in range(n_batches):
            
            a_batch = a_batches[j]
            t_batch = t_batches[j]
            F_batch = F_batches[j]
            
            f = lambda t, k: spring_mass_system(t, m=m, c=c, k=k, F=F_batch, x0=x0_batch, xdot0=v0_batch, dt=dt)
            f1 = lambda t, k: spring_mass_system1(t, m=m, c=c, k=k, F=F_batch, x0=x0_batch, xdot0=v0_batch, dt=dt)
            
            try:
                k_param, _ = curve_fit(f, t_batch, a_batch, bounds=k_bounds)
                k_param = k_param[0]
            except(RuntimeError): # If optimal parameter is not found and error is thrown
                k_param = k_bounds[0]
            
            reconstructed_data = f1(t_batch, k_param)
            a_r = reconstructed_data[:,0]
            v_r = reconstructed_data[:,1]
            x_r = reconstructed_data[:,2]
            
            data_pred[0, N*j:N*(j+1)] = k_param
            data_pred[1, N*j:N*(j+1)] = a_r
            
            x0_batch = x_r[-1]
            v0_batch = v_r[-1]
        
        
        data_pred_tot[test] = data_pred
        print('finished test #%d'%(test+1))
    np.save('./model_predictions/pure_physics/k_pred.npy', data_pred_tot)
    #%%

if __name__ == '__main__':
    main()