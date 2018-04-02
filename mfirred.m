function [rep, T] = mfirred(H, l, H0, n)
% function [rep, T] = mfirred(H, l, H0, n)
% 
% Create a balanced truncation of a multivariate FIR
% realization. See mfir() for detils on the "canonical"
% representation. The addition here is to create
% the balancing transformation given the Gramian P
% assuming that Q=I, and then truncating the transformed
% state to dimension n (where 1 <= n <= l * ny).
%
% If n is not provided; the full balanced realization is
% returned. If H0 is not provided or is empty [] then
% the direct term will be zero. The full state 
% balancing transform is optionally returned as T;
% x = T * xbal.
%

if nargin < 3
  H0 = [];
end

[A, B, C, D, P] = mfir(H, l, H0);

ny = size(C, 1);
nu = size(B, 2);
nx = size(A, 1);

assert(nx == l * ny);

if nargin < 4
  n = nx;
end

assert(n >= 1 && n <= nx)

rep = struct;

[U, S, V] = svd(P);
rep.sv = sqrt(diag(S));

ssv = sqrt(rep.sv);
T = U(:, 1:n) * diag(ssv(1:n));
Ti = diag(1./ssv(1:n)) * (U(:, 1:n)');

rep.A = Ti * A * T;
rep.B = Ti * B;
rep.C = C * T;
rep.D = D;

if nargout > 1
  T = U * diag(ssv);
end

end
