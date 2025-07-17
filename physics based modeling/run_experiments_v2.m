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
n_tests = 100;

with_friction = true;
% friction element
F_brk = 0.05;
v_brk = 0.01;
F_c = 0.9*F_brk;

% time point range of first line segment end (as a ratio of test length)
x_range1 = [0.25, 0.45];
x_range2 = [0.55, 0.75];
% percentage of possible degredation at time point 1 and 2
y_range1 = [.57, .85];
y_range2 = [.14, .43];
%%
n_timesteps = floor(test_length/step_size);
t = linspace(0, test_length, n_timesteps)';
F = sin(2*pi*freq*t);
F_signal = [t, F];
%%
for i = 1:n_tests
    %%
    % produce k
    x1 = x_range1(1) + rand()*(x_range1(2)-x_range1(1));
    x2 = x_range2(1) + rand()*(x_range2(2)-x_range2(1));
    y1 = y_range1(1) + rand()*(y_range1(2)-y_range1(1));
    y2 = y_range2(1) + rand()*(y_range2(2)-y_range2(1));

    y_interp = [1, 1, y1, y2, 0, 0];
    x_interp = [0, 0.15, x1, x2, 0.85, 1];
    y_interp = (starting_k - stopping_k)*y_interp + stopping_k;
    x_interp = x_interp*test_length;
    k = interp1(x_interp, y_interp, t);
    k_signal = [t, k];
    %%
    if with_friction
        out = sim('degrade_with_friction_v2.slx');
    else
        out = sim('degrade_no_friction_v2.slx');
    end
    save_data_out(out, i, with_friction);
    %%
end
function save_data_out(out, i, with_friction)
    x = out.displacement(:,2);
    v = out.velocity(:,2);
    a = out.acceleration(:,2);
    t = out.velocity(:,1);
    F = out.force(:,2);
    k = out.stiffness(:,2);
    if any(isnan(a))
        fprintf('nan values found.')
    end
    arr = [t, x, v, a, k, F];
    if with_friction
        writematrix(arr, "./data/with_friction/test_"+i+".csv")
    else
        writematrix(arr, "./data/no_friction/test_"+i+".csv");
    end
end