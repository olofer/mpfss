%
% SCRIPT: test_pdipmqpneq3
%
% NOTE: tested with OCTAVE (4.2.0 + optim pkg) only!
%

n = 250;
% Random objective Hessian and random linear term
M = randn(n);
H = M'*M;
h = randn(n, 1);
% Require upper or lower bound for some variables
E = [eye(n); -eye(n)];
f = 10 * [ones(n, 1); ones(n, 1)];

nh = ceil(n/2);
idx = randperm(n);
idx = idx(1:nh)';
E = E(idx,:);
f = f(idx);

% Solve; min_z { 0.5*z'*H*z + h'*z }, s.t. E*z <= f
fprintf(1, '[%s]: *** test problem ***\n%i decision variables\n%i inequality constraints.\n', ...
  mfilename(), size(H, 1), size(E, 1));

%
% Solve with pdipmqpneq3()
%

t0 = tic;
rep0 = pdipmqpneq3(H, h, E, f, 100, 1e-10, 0.96);
t0 = toc(t0);

if ~rep0.isconverged
  fprintf(1, '%s did not converge!\n', rep0.solver);
else
  disp([rep0.solver, ' => f0*=', num2str(rep0.fx, 15), ...
    '; #iters=', num2str(rep0.iters), '; clock=', num2str(t0),' sec.']);
end

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
  fprintf(1, 'Running OCTAVE.\n');
  pkg load optim; % try to load the optim package to access MATLAB-like quadprog()
else
  fprintf(1, 'Running MATLAB.\n');
  assert(license('test', 'optimization_toolbox') == 1); % check license for required toolbox
end
  
%
% Then obtain a "reference" result from OCTAVE or MATLAB
%

if ~isOctave
  % MATLAB
  ref_solver_name = sprintf('matlab::%s', 'quadprog()');
  optopts = optimset(...
    'Algorithm', 'interior-point-convex', ...
    'Display', 'off', 'TolX', 1e-12, 'TolFun', 1e-12);
  t1 = tic;
  [xqp1, fqp1, exitqp, outqp] = quadprog(H, h, E, f, [], [], [], [], [], optopts);
  t1 = toc(t1);
else
  % OCTAVE
  ref_solver_name = sprintf('octave::%s', 'quadprog()');
  t1 = tic;
  [xqp1, fqp1, exitqp, outqp] = quadprog(H, h, E, f, [], [], [], [], []);
  t1 = toc(t1);
end

if exitqp ~= 1
  fprintf(1, '%s did not converge!\n', ref_solver_name);
else
  disp([ref_solver_name, ' => f1*=', num2str(fqp1, 15), ...
    '; #iters=', num2str(outqp.iterations), '; clock=', num2str(t1), ' sec.']);
end

% Compare solutions
twonorm_rel_err = norm(xqp1-rep0.x, 2)/norm(xqp1, 2);
disp(['norm(x0*-x1*)/norm(x1*) = ', num2str(twonorm_rel_err, 15)]);
if twonorm_rel_err < 1e-7
  fprintf(1, '[%s]: looks OK.\n', mfilename());
else
  fprintf(1, '[%s]: does NOT look OK.\n', mfilename());
end
