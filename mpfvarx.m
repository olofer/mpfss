function rep = mpfvarx(DB, ords, dterm, autoscl)
% function rep = mpfvarx(DB, ords, dterm, autoscl)
%
% Estimate state-space system representation
% directly from vector autoregressive (VARX)
% block coefficients and balanced truncation.
%
% DB is a cell array of input-output data
% (vector time-series u, y; samples in columns).
%
% ords = [p n] = [lag order] determines
% what size n the estimated system should
% be delivered with; and the VARX lag length p.
% If the output signal has dim. ny then the 
% maximal state n is equal to ny * p.
%
% dterm = 1 if direct feedthrough is part 
% of the system; otherwise put dterm = 0.
%
% autoscl = 1 to numerically condition
% the I/O signals before estimating VARX.
%
% Detrending or scaling is the 
% responsibility of the caller; no
% special preprocessing is done by
% this routine.
%

%
% TODO: one "obvious" improvement to this
% code is to support regularized VARX estimation
% which might significantly up the accuracy on 
% smaller / marginally rich I/O datasets.
%
% TODO: another "obvious" improvement is to 
% use better model reduction of the VARX system
% (the information is there but the truncation
%  need to be frequency weighted / subspace adapted)
%

% Quick (incomplete) input sanity check:
assert(nargin >= 3, 'must provide at least 3 inputs');
assert(iscell(DB), 'DB must be a cell array');
assert(numel(DB) >= 1, 'DB cannot be empty');
assert(numel(ords) == 2, 'ords must have 2 elements');
p = ords(1);
n = ords(2);
ny = size(DB{1}.y, 1);
nu = size(DB{1}.u, 1);
assert(p >= 1);
assert(n >= 1 && n <= p * ny);
assert(numel(dterm) == 1);
assert(dterm == 0 || dterm == 1);

if nargin <= 3
  autoscl = 1;  % default is to autoscale signals (globally)
end

assert(autoscl == 0 || autoscl == 1);

rep = struct;
rep.ords = ords;
rep.dterm = dterm;
rep.autoscl = autoscl;

% (optional) Step 0: scale signals to unity RMS
% (better numerical condition)
rmsy = 1; rmsu = 1;
if autoscl == 1
  [Ryy, Ruu] = get_yu_cov(DB);
  rmsy = sqrt(trace(Ryy) / ny);
  rmsu = sqrt(trace(Ruu) / nu);
end

% Step 1: estimate VARX block coefficients
scl_yu = [1 / rmsy, 1 / rmsu];
[Ghat, ntot] = batchdvarxestcov(DB, p, dterm, scl_yu);
assert(size(Ghat, 1) == ny);

rep.rmsyu = [rmsy, rmsu];
rep.ntot = ntot;  % total number of time-stamps used.
rep.spp = ntot / (p * ny);  % samples per parameter

% Ghat are the VARX blocks [H(1)...H(p)] each 
% block of dim ny-by-(nu+ny). If dterm == 1 then
% to Ghat is also appended the block D
% (rightmost nu columns).
if dterm == 0
  H = Ghat;
  H0 = [];
else
  H = Ghat(:, 1:((nu + ny) * p));
  D = Ghat(:, (1 + (nu + ny) * p):end);
  assert(size(D, 2) == nu && size(D, 1) == ny);
  H0 = [D, zeros(ny, ny)];
end

% Step 2: create / truncate a predictor form representation
% with [u; y] as input and yhat as output; VFIR system
% so always stable and Gramians exist.
tmp = mfirred(H, p, H0, n);

rep.H = H;
rep.H0 = H0;

% Step 3: pull out the system (A,B,C,D) from the 
% reduced/stable predictor state space form.
% Return data in output struct rep.
rep.K = tmp.B(:, (nu+1):end);
rep.D = tmp.D(:, 1:nu) * (rmsy / rmsu);
rep.C = tmp.C;
rep.B = (tmp.B(:, 1:nu) + rep.K * tmp.D(:, 1:nu)) * (rmsy / rmsu);
rep.A = tmp.A + rep.K * rep.C;

assert(size(rep.A, 1) == n);
assert(size(rep.B, 1) == n);
assert(size(rep.K, 1) == n);

end

function [Ryy, Ruu] = get_yu_cov(DB)
Ryy = DB{1}.y * DB{1}.y';
Ruu = DB{1}.u * DB{1}.u';
nnb = size(DB{1}.y, 2);
for bb = 2:numel(DB)
  Ryy = Ryy + DB{bb}.y * DB{bb}.y';
  Ruu = Ruu + DB{bb}.u * DB{bb}.u';
  nnb = nnb + size(DB{bb}.y, 2);
end
Ryy = (1/nnb) * Ryy;
Ruu = (1/nnb) * Ruu;
end

function [Ghat, nt] = batchdvarxestcov(DB, p, dterm, scl_yu)
% General estimation of vector autoregressive models (VARXs)
% with optional direct term D from input-output data;
% VARX order is p (=na=nb); does batch-by-batch squaring.
% Standardised least-squares estimation; Y = G*Z + E
nb = numel(DB);
[Yb, Zb] = dvarxdata(DB{1}.y, DB{1}.u, p, dterm, scl_yu);
nt = size(DB{1}.y, 2);
YZt = Yb * Zb';
ZZt = Zb * Zb';
for bb = 2:nb
  [Yb, Zb] = dvarxdata(DB{bb}.y, DB{bb}.u, p, dterm, scl_yu);
  YZt = YZt + Yb * Zb';
  ZZt = ZZt + Zb * Zb';
  nt = nt + size(DB{bb}.y, 2);
end
Ghat = YZt / ZZt;
end

% Create regressor for one contiguous batch of time-series data
function [Y, Zp] = dvarxdata(y, u, p, dterm, scl_yu)
ny = size(y, 1);
nu = size(u, 1);
N = size(y, 2);
k1 = p + 1;
k2 = N;
Neff = k2 - k1 + 1;
nz = ny + nu;
Z = [u * scl_yu(2); y * scl_yu(1)];
Y = zeros(ny, Neff);
nzp = nz * p;
if dterm > 0  % augment with direct term
  Zp = zeros(nzq + nu, Neff);
  for k = k1:k2
    kk = k - k1 + 1;
    Y(:, kk) = y(:, k) * scl_yu(1);
    Zp(:, kk) = [reshape(Z(:, (k-1):-1:(k-p)), nzp, 1); u(:, k) * scl_yu(2)];
  end
else  % no direct term
  Zp = zeros(nzp, Neff);
  for k = k1:k2
    kk = k - k1 + 1;
    Y(:, kk) = y(:, k) * scl_yu(1);
    Zp(:, kk) = reshape(Z(:, (k-1):-1:(k-p)), nzp, 1);
  end
end
end
