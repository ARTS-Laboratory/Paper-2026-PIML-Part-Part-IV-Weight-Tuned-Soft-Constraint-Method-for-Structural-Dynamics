% 120 second tests, all with 5 Hz excitation
% The path of degradation will change for each test
% 100 tests total
starting_k = 1500;
stopping_k = 500;
step_size = 0.001;
test_length = 120;
m=1;
c=.2;
freq = 5;

% friction element
% F_brk = 0.05;
% v_brk = 0.01;
% F_c = 0.9*F_brk;

% run simulink-simscape experiment
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(freq*2*pi*t);
F_signal = [t, F];

y_interp = [starting_k, starting_k, stopping_k, stopping_k];
x_interp = [0, 0.15*test_length, 0.85*test_length, test_length];

k = interp1(x_interp, y_interp, t);
k_signal = [t, k];

out = sim('pinn_degrade_v3.slx');

% save data
x = out.displacement(:,2);
v = out.velocity(:,2);
a = out.acceleration(:,2);
j = out.jerk(:,2);
t = out.velocity(:,1);
F = out.force(:,2);
k = out.stiffness(:,2);

% plot to see if residual is consistent.
r = F - m*a - c*v - k.*x;

figure
plot(t, r, t, k.*x)

Fdot = 2*pi*freq*cos(2*pi*freq*t);
Fddot = -(2*pi*freq)^2*sin(2*pi*freq*t);

arr = [t, x, v, a, j, k, F, Fdot, Fddot];
writematrix(arr, "./data/pinn_data/test_"+0+".csv");