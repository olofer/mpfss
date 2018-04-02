function [P, bal] = test_mfir(ny, nu, l, r)
% function [P, bal] = test_mfir(ny, nu, l, r) 
%
% Checks intended functionality of the mfir() program. 
% Generates random FIR block coefficient with the specified
% size ny-by-nu and the specified lag length l.
%
% All error print outs should be zero.
%
% Balanced truncation of the FIR representation will also
% be checked if the r argument is provided (state order r).
%

H = randn(ny, l * nu);
H0 = randn(ny, nu);
[A, B, C, D, P] = mfir(H, l, H0);

nd = size(A, 1);
assert(nd == ny * l);

% Extract impulse response by simulation and check against H exactly;
fprintf(1, 'norm(H)  = %e\nnorm(H0) = %e\n', norm(H, 'fro'), norm(H0, 'fro'));

% Need to commit to a specific component for the impulse:
for uu = 1:nu
  u = zeros(nu, 1);
  u(uu) = 1;
  Hout = [];
  x = zeros(ny * l, 1);
  y0 = C * x + D * u;
  x = A * x + B * u;
  for ii = 1:l
    Hout = [Hout, C * x];
    x = A * x;
  end
  fprintf(1, 'd (%i) error = %e\n', uu, norm(H0 * u - y0, 'fro'));
  fprintf(1, 'h (%i) error = %e\n', uu, norm(H(:, uu:nu:end)- Hout, 'fro'));
end

% Next check the Gramian P is correct by explicit (inefficient) calc.
cum = zeros(nd, nd);
tmp = B * B';
for ii = 1:l
  cum = cum + tmp;
  tmp = A * (tmp * A');
  if ii == l
    fprintf(1, 'P:tmp[%i] = %e\n', ii, norm(tmp, 'fro'));
  end
end
% this may be non-zero but ~1e-14 or so due to finite precision ?
fprintf(1, 'norm(P)  = %e\n', norm(cum, 'fro'));
fprintf(1, 'error(P) = %e\n', norm(cum - P, 'fro'));

% Confirm that the Gramian Q is I
cum = zeros(nd, nd);
tmp = C' * C;
for ii = 1:l
  cum = cum + tmp;
  tmp = A' * (tmp * A);
  if ii == l
    fprintf(1, 'Q:tmp[%i] = %e\n', ii, norm(tmp, 'fro'));
  end
end
fprintf(1, 'norm(Q)  = %e\n', norm(eye(nd), 'fro'));
fprintf(1, 'error(Q) = %e\n', norm(cum - eye(nd), 'fro'));

% Basic optional check of balanced truncation code
bal = [];
if nargin > 3
  [bal, T] = mfirred(H, l, H0, r);
  % Check that T actually generates balanced Gramians P, Q
  Pbal = T \ (P / (T'));
  Qbal = T' * T;
  Sbal = diag(Qbal);
  % Check how far away from the diagonal target the balanced Gramians are.
  norm_Qbal = norm(Qbal, 'fro');
  fprintf(1, 'rel err Qbal = %e\n', norm(diag(Sbal) - Qbal, 'fro') / norm_Qbal);
  norm_Pbal = norm(Pbal, 'fro');
  fprintf(1, 'rel err Pbal = %e\n', norm(diag(Sbal) - Pbal, 'fro') / norm_Pbal);
  fprintf(1, 'rel diff Pbal,Qbal = %e\n', norm(Pbal - Qbal, 'fro') / norm_Qbal);
  % (print outs should be small 1e-14 , 1e-15 type numbers)
end

end
