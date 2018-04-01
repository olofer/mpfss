function [A, B, C, D, P] = mfir(H, l, H0)
% function [A, B, C, D, P] = mfir(H, l, H0)
%
% Create a MIMO FIR state-space representation given
% block FIR coefficients: H = [H1, H2, H3, ..., Hl]
% and the optional direct term block H0; the output
% size ny is automatically determined by the row count
% of H and the input size nu is determined by the column
% count of H and the lag length l.
% H0 can be omitted or empty [] or otherwise must be
% of size ny-by-nu.
%
% State-space representation (A,B,C,D) is returned
% with the state size equal to ny * l.
%
% The controllability matrix P is optionally
% returned and is useful for model reduction.
% The observability matrix is just unity.
%
% Properties are checked by test_mfir() companion.
%

assert(l > 0);
ny = size(H, 1);
nu = round(size(H, 2) / l);
assert(nu * l == size(H, 2), ...
  'ncol(H) with given lag l does not make sense');

nd = ny * l;
A = zeros(nd, nd);
B = zeros(nd, nu);
C = zeros(ny, nd);
D = zeros(ny, nu);

rr = 0;
cc = 0;
for ii = 1:l
  idxr = (rr + 1):(rr + ny);
  idxc = (cc + 1):(cc + nu);
  B(idxr, :) = H(:, idxc);
  if ii ~= l
    A(idxr, idxr + ny) = eye(ny);
  end
  rr = rr + ny;
  cc = cc + nu;
end

C(:, 1:ny) = eye(ny);

if (nargin > 2) && ~isempty(H0)
  assert(size(H0, 1) == ny);
  assert(size(H0, 2) == nu);
  D(:, :) = H0;
end

if nargout > 4
  % Compute matrix P also
  P = zeros(nd, nd);
  V = B;
  for ii = 1:l
    P = P + V * V';
    V = [V((ny + 1):nd, :); zeros(ny, nu)];
  end
end

end
