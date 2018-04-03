function [dat, rep] = test_mpfss_openloop(M, ny, nu, N, n)
%
% function [dat, rep] = test_mpfss_openloop(M, ny, nu, N, n) 
%
% Demonstration of n-state subspace system identification.
% "True" data-generating plant has M modes (truncated infinite sequence)
% based on a PDE describing a damped vibrating string.
%
% System description found here:
%   Olofsson and Rojas,  16th IFAC Symposium on System Identification, pp 362-367, 2012
%   DOI: https://doi.org/10.3182/20120711-3-BE-2027.00167
%
% There are ny outputs and nu inputs.
% N samples (u, y) will be generated (returned in struct dat).
% The estimated system data will be returned in struct rep.
%
% Three plots will be generated:
% (i) output time traces
% (ii) sigma plot of true vs. estimated transfer function
% (iii) pole plot showing eigenvalues for true and estimated systems
%
% DEFAULT USAGE: test_mpfss_openloop();
%
% Tested with OCTAVE 4.2.0
%

if nargin < 5
  n = 20; % 10 rotational modes max as default
end

if nargin < 4
  N = 5000;
end

if nargin < 3
  nu = 2;
end

if nargin < 2
  ny = 8;
end

if nargin < 1
  M = 100;
end

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
  pkg load control; % need to load this package for OCTAVE
else
  % Check for MATLAB control toolbox license
  assert(license('test', 'control_toolbox') == 1);
end

L = 5 / 11;
c = 400;
tau = L / c;
Ts = (1/50) * tau;
r1 = 0.1 * tau;
r2 = 50 / (c * L);

% quadrature integration accuracy
qtol = 1e-12;

% Allocate system matrices (continuous time)
A = zeros(2*M, 2*M);
B = zeros(2*M, nu);
C = zeros(ny, 2*M);

% Generate cell array of input field shape functions
fu = cell(nu, 1);
for ii = 1:nu
  xplonk = (L / 16) + (L/(nu + 1)) * (ii - 1);
  assert(xplonk < L && xplonk > 0);
  fu{ii} = @(x) (exp(-(x-xplonk)^2/(0.01*L)^2));
end

% Pick-up line coordinates
rho = 1;
xpick = (1:ny) * rho * L / (ny + 1);

% Populate (A, B, C)
for mm = 1:M
  kk = 2 * (mm - 1) + 1;
  A(kk:(kk+1),kk:(kk+1)) = [0, 1; -(c*mm*pi/L)^2, -(r1 + r2*(mm*pi/L)^2)];
  Xm = @(x)(sin(x*pi*mm/L));
  % B matrix entries
  for ii = 1:nu
    fi = fu{ii};
    tmp = quad(@(x)(fi(x).*Xm(x)), 0, L, qtol);
    B(kk:(kk+1), ii) = [0; (2 / L) * tmp];
  end
  % C matrix entries
  for ii = 1:ny
    C(ii, kk:(kk+1)) = [Xm(xpick(ii)), 0];
  end
end

% Store original CT system in struct and copy it into the return struct rep below
sys0 = struct;
sys0.A = A;
sys0.B = B;
sys0.C = C;

sysdt = c2d(ss(A, B, C, 0), Ts, 'zoh');
sysdt = ss(sysdt.A, sysdt.B, sysdt.C, 0, -1); % remove Ts from DT system

dat = cell(1, 1);

U = randn(N, nu);
[Y, T] = lsim(sysdt, U);
stdy = sqrt(trace(Y'*Y)/(ny*N));
Y = Y + 0.05*stdy*randn(N, ny);
dat{1}.u = U';
dat{1}.y = Y';

figure;
plot(T, Y);
xlabel('timestep');
ylabel('pickup array y');
title(sprintf('Ts = %f ms; stdy=%e', Ts * 1e3, stdy));

% ***********************************************
% HERE IS THE SUBSPACE SYSTEM IDENTIFICATION CALL
ords = [10 20 20 n]; % retain n states
dterm = 0;
rep = mpfssvarxbatchestlm(dat, ords, dterm);
% alternative approach using direct VARX then weighted
% model reduction; works good too!
rep2 = mpfvarx(dat, [30 n], dterm);
% ***********************************************

% Add sys0 to output struct
rep.sys0 = sys0;

syses = ss(rep.A, rep.B, rep.C, 0, -1);
syses2 = ss(rep2.A, rep2.B, rep2.C, 0, -1);

figure;
sigma(sysdt, 'b-', syses, 'r-', syses2, 'g-');

% Finally plot the eigenvalues also
figure;
th = linspace(0, 2*pi, 500);
plot(cos(th), sin(th), 'k-');
hold on;
E0 = eig(sysdt.A);
plot(real(E0), imag(E0), 'bx');
E1 = eig(rep.A);
plot(real(E1), imag(E1), 'ro');
E2 = eig(rep2.A);
plot(real(E2), imag(E2), 'gd');
axis equal;
xlabel('Re(eig)');
ylabel('Im(eig)');
title('blue = truth, red = estim., green=estim.alt.');

end
