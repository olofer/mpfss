function rep = gen_dense_lti_qp_abc(A, B, C, Qx, Qu, Qy, n)
% function rep = gen_dense_lti_qp_abc(A, B, C, Qx, Qu, Qy, n)
%
% Generate quadratic program (QP) objective data for the model predictive
% control (specifically: condensed MPC) problem:
%
% min sum_{i=1}^{n} (1/2) * { x(i)'*Qx*x(i) + u(i-1)'*Qu*u(i-1) + e(i)'*Qy*e(i)  }
%
% s.t. x(i+1) = A*x(i) + B*u(i) + w(i)
%      e(i) = y(i) - r(i)
%
% w(i), i = 0..n-1 is a (optionally provided) bias signal
% u(i), i = 0..n-1 is the control signal to be solved for
% x(i), i = 0..n is the system state; where x(0) is provided
% y(i), i = 1..n is the system output signal
% r(i), i = 1..n is the (optionally provided) output reference signal
%
% The projection matrices needed to instantiate the QP are
% returned in the return struct rep.
%
% With the above setup. The output struct members are used as follows.
% Let X be a stacking of vectors x(i), i = 1..n then
%
% X = P0*x(0) + Pu*U + Pw*W
%
% where U is a stacking of u(i), and W is stacking of w(i), both i = 0..n-1;
% Here (block-structured) P0, Pu, and Pw are members of rep.
% Also let R be a stacking of r(i), i = 1..n, then the QP cost data (H, h, h0)
% in the sense; 0.5*U'*H*U + U'*z + h0, can be instantiated from x(0), W, R 
% as follows.
%
% H = Hu + Pu'*(Hx+Cx'*Hy*Cx)*Pu
% h = M0'*x(0) + Mw'*W + Mr'*R
% h0 = (not writing out the mess)
%
% where Hu = is blkdiag(Qu), Hx = blkdiag(Qx),
% Hy = blkdiag(Qy), and Cx = blkdiag(C)
%
% and where the required matrices are fields in the return struct rep:
%   rep.{M0, Mw, Mr, H}
%

rep = struct;

assert(n >= 1);

nx = size(A, 1);
assert(size(A, 2) == nx);
nu = size(B, 2);
assert(size(B, 1) == nx);
ny = size(C, 1);
assert(size(C, 2) == nx);

Qx = prepare_sym_matrix(Qx, nx);
Qu = prepare_sym_matrix(Qu, nu);
Qy = prepare_sym_matrix(Qy, ny);

% Allocate and populate the "projection" matrices
rep.P0 = zeros(nx * n, nx);
rep.Pu = zeros(nx * n, nu * n);
rep.Pw = zeros(nx * n, nx * n);

rep.P0(1:nx, :) = A;
for ii = 1:(n-1)
  rep.P0(ii * nx + (1:nx), :) = A * rep.P0((ii - 1) * nx + (1:nx), :);
end

rep.Pu(1:nx, 1:nu) = B;
for ii = 1:(n-1)
  rep.Pu(ii * nx + (1:nx), 1:nu) = A * rep.Pu((ii - 1) * nx + (1:nx), 1:nu);
  rep.Pu(ii * nx + (1:nx), (nu + 1):(n * nu)) = rep.Pu((ii - 1) * nx + (1:nx), 1:(n * nu - nu));
end

rep.Pw(1:nx, 1:nx) = eye(nx);
for ii = 1:(n-1)
  rep.Pw(ii * nx + (1:nx), 1:nx) = A * rep.Pw((ii - 1) * nx + (1:nx), 1:nx);
  rep.Pw(ii * nx + (1:nx), (nx + 1):(n * nx)) = rep.Pw((ii - 1) * nx + (1:nx), 1:(n * nx - nx));
end

% Generate the cost matrix specifiers
% (this could be optimized but it does not matter for typical use-cases)
% (TODO: might want to expand this to allow discounting factor on diagonal)
Hx = kron(eye(n), Qx);
Hu = kron(eye(n), Qu);
Hy = kron(eye(n), Qy);
Cx = kron(eye(n), C);
Hx = Hx + Cx'*Hy*Cx;

% Final matrices of interest 
rep.H = Hu + rep.Pu'*Hx*rep.Pu;
rep.M0 = rep.P0'*Hx*rep.Pu;
rep.Mw = rep.Pw'*Hx*rep.Pu;
rep.Mr = -Hy*Cx*rep.Pu;

% TODO: optionally generate the matrices needed to form the constant
% h0 per instance of (x0, w, r) also ...

end

function M = prepare_sym_matrix(M, m)
  if size(M, 1) == m && size(M, 2) == m
    % already a square matrix but maybe not symmetric
    L = tril(M, -1); % force copy of lower triangle to upper triangle
    D = diag(M);
    M = L + L' + diag(D); % guaranteed to be symmetric
  elseif size(M, 1) == m && size(M, 2) == 1
    M = diag(M(:));
  elseif size(M, 1) == 1 && size(M, 2) == 2
    M = diag(M(:));
  elseif numel(M) == 1
    M = eye(m) * M(1);
  else
    error('argument dimension makes no sense');
  end
  % No check for pos. def
end
