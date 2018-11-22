function [dat, rep] = test_mpfss(num_batches, samples_per_batch)
%
% function [dat, rep] = test_mpfss(num_batches, samples_per_batch)
%
% Simulates an unstable 2-by-2 MIMO LTI system in closed-loop
% with a simple non-linear (constrained) predictive controller.
%
% Generates batches of data and passes the full set to the 
% subspace system identification code.
%
% The resulting LTI system estimate is then shown together
% with the underlying true system in a sigma plot.
%
% The time-series samples for the individual batches
% will be plotted if no outputs arguments are specified.
%
% Call without second output to only generate the dataset.
%
% NOTE: tested with both OCTAVE (4.2.0) and MATLAB (R2016b).
%

makeBatchFigures = (nargout == 0);

if nargin < 2
  samples_per_batch = 500;
end

if nargin < 1
  num_batches = 1;
end

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
  % https://wiki.octave.org/Control_package
  pkg load control; % need to load this package for OCTAVE
else
  % (optional) TODO: check for MATLAB control toolbox license
  assert(license('test', 'control_toolbox') == 1);
end

%
% LTI system definition and discretization 
%

Ac=[-.151,-60.5651,0,-32.174;
    -.0001,-1.3411,.9929,0;
    .00018,43.2541,-.86939,0;
    0,0,1,0];

Bc=[-2.516,-13.136;
    -.1689,-.2514;
    -17.251,-1.5766;
    0,0];

Cc=[0,1,0,0;
    0,0,0,1];

Ts = 0.05;
sysd = c2d(ss(Ac, Bc, Cc, 0), Ts, 'zoh');

nx = size(Ac, 1);
nu = size(Bc, 2);
ny = size(Cc, 1);

fprintf(1, 'max(abs(eig(A))) = %.4f\n', max(abs(eig(sysd.A))));

%
% Generation of several closed loop simulations of the above system 
%

dat = cell(1, num_batches);
n = samples_per_batch;
nhorz = 10;

fprintf(1, ...
  '[%s]: %i closed-loop control simulations (each with %i time steps)...\n', ...
  mfilename(), num_batches, n);

for bb = 1:num_batches
  % Simulate a single closed-loop experiment
  [u, y, x, r] = closed_loop_sim_mpc(sysd, n, nhorz);
  
  % Store batch in cell array; no access to state x
  eb = struct;
  eb.u = u;
  eb.y = y;
  eb.r = r;
  eb.t = Ts * (0:(n - 1))';
  dat{bb} = eb;
  
  % Show progress
  fprintf(1, '[%s]: batch #%4i / %i\n', mfilename(), bb, num_batches);
  
  if makeBatchFigures
    figure;
    subplot(2, 1, 1);
    stairs(dat{bb}.t, dat{bb}.u');
    ylabel('controls u');
    subplot(2, 1, 2);
    stairs(dat{bb}.t, dat{bb}.y');
    hold on;
    stairs(dat{bb}.t, dat{bb}.r', 'k')
    ylabel('outputs y');
    xlabel(sprintf('batch #%i time [sec]', bb));
  end
end

if nargout < 2
  return;
end

% Automatically call the subspace sysid code; process all batches into a single system estimate

% ***********************************************
% HERE IS THE SUBSPACE SYSTEM IDENTIFICATION CALL
ords = [10 20 20 4]; % retain 4 states
dterm = 0;
rep = mpfssvarxbatchestlm(dat, ords, dterm);
% ***********************************************

%
% TODO: maybe do auto model reduction instead of fixed 4 states
% (balanced realization, based on finite-time Gramians maybe)
%

figure;
sigma(ss(sysd.A, sysd.B, sysd.C, 0, -1), 'b-', ss(rep.A, rep.B, rep.C, 0, -1), 'r-');
legend('Actual system', 'Estimated system');
title('Open-loop system singular values');

end

%
% Sub-program to generate a single dataset (u, y)
%

function [u, y, x, r, condenseData] = closed_loop_sim_mpc(sysd, nstep, nhorz)

assert(nstep > 0 && nhorz > 0);

Qx = 0;
Qu = 1e-3;
Qy = 1.0;

condenseData = gen_dense_lti_qp_abc(sysd.A, sysd.B, sysd.C, Qx, Qu, Qy, nhorz);

nx = size(sysd.A, 1);
nu = size(sysd.B, 2);
ny = size(sysd.C, 1);

stde = 0.01;
stdw = 0.10;
umax = 25;
ymax = 5; %10;

switchInterval = 100;

F2 = [eye(nu);-eye(nu)];
f3 = umax * ones(2 * nu, 1);
E = kron(eye(nhorz), F2);
f = kron(ones(nhorz, 1), f3);

u = zeros(nu, nstep);
y = zeros(ny, nstep);
x = zeros(nx, nstep);
r = zeros(ny, nstep);

X = randn(nx, 1);
W = zeros(nx * nhorz, 1);
assert(ny == 2);
R = zeros(ny * nhorz, 1);

for kk = 1:nstep
  % Store current state and output
  e = stde * randn(ny, 1);
  Y = sysd.C * X + e;
  y(:, kk) = Y;
  x(:, kk) = X;
  % Decide what the target output vector should be
  if rem(kk, switchInterval) == 0
    R = kron(ones(nhorz, 1), (2 * rand(ny, 1) - 1) * ymax );
  end
  r(:, kk) = R(1:ny);
  % Solve for control action uc; ASSUMING perfect knowledge of
  % current state X and system (A, B, C)
  h = condenseData.M0'*X + condenseData.Mr'*R; 
  qpReport = pdipmqpneq3(condenseData.H, h, E, f);
  uc = qpReport.x(1:nu);
  u(:, kk) = uc;
  % Evolve system one time step
  w = stdw * randn(nx, 1);
  X = sysd.A * X + sysd.B * uc + w;
end

end
