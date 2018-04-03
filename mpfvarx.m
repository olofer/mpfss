function rep = mpfvarx(DB, ords, dterm, autoscl, cholla)
% function rep = mpfvarx(DB, ords, dterm, autoscl, cholla)
%
% Estimate state-space system representation
% directly from vector autoregressive (VARX)
% block coefficients followed by weighted
% truncation of the predictor form.
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
% cholla >= 0 enables an alternative
% calculation of the truncating transform.
% (default is cholla < 0)
%
% Detrending / conditioning is the 
% responsibility of the caller; 
% rudimentary I/O scaling is 
% performed automatically.
%

%
% TODO: one "obvious" improvement to this
% code is to support regularized VARX estimation
% which might significantly up the accuracy on 
% smaller / marginally rich I/O datasets.
%

%
% TODO: more extensive benchmarking of this simpler
% approach compared to the subspace version; both 
% should be able to do open- and closed loop but which
% one is quicker (this), more accurate (?!), easier (this) ...
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

if nargin < 4
  autoscl = 1;  % default is to autoscale signals (globally)
end

assert(autoscl == 0 || autoscl == 1);

if nargin < 5
  cholla = -1;  % default is to not use Cholesky weighted numerics
end

assert(numel(cholla) == 1 && isfinite(cholla));

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
[Ghat, ntot, ZZt] = batchdvarxestcov(DB, p, dterm, scl_yu);
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
  % The last nu rows and columns must be cut away
  % prior to the model reduction step below.
  ZZt = ZZt(1:((nu + ny) * p), 1:((nu + ny) * p));
end

% Step 2: create / truncate a predictor form representation
% with [u; y] as input and yhat as output; 
% Do not use the "balanced truncation" function; rather
% do "weighted" truncation as follows.

rep.H = H;
rep.H0 = H0;

[Apf, Bpf, Cpf, Dpf] = mfir(H, p, H0);
Mpf = zeros(p * ny, p * (ny + nu));  % allocate input to state map
Mpf(:, 1:(ny + nu)) = Bpf;
cc = ny + nu;
for ii = 2:p
  % due to structure of Apf; the next line could be optimized by
  % (block) shifting the previous columns in Mpf (to be done).
  %Mpf(:, (cc + 1):(cc + ny + nu)) = Apf * Mpf(:, (cc - ny - nu + 1):cc);

  % NOTE: there will be many zeros "pointlessly" assigned
  % so the below line can still be improved
  Mpf(1:((p - 1) * ny), (cc + 1):(cc + ny + nu)) = Mpf((ny + 1):end, (cc - ny - nu + 1):cc);
  cc = cc + ny + nu;
end
assert(cc == p * (ny + nu));

if cholla < 0
  % Construct a weighted PCA-like transformation
  P = (Mpf * (ZZt / ntot)) * Mpf';  % weighted "Gramian"
  [Up, Sp, Vp] = svd(P);  % square decomp.
  rep.sv = sqrt(diag(Sp));
  T = Up(:, 1:n) * diag(rep.sv(1:n));
  Ti = diag(1./rep.sv(1:n)) * Up(:, 1:n)';
else
  % alternative numerics; use with cholla = 0 for equivalence 
  % to the standard code above; cholla > 0 allows "interpolation"
  % between "standard" and "unweighted" (cholla very large)
  L = chol(ZZt / ntot + cholla * eye(p * (nu + ny)), 'lower');
  [Ul, Sl, Vl] = svd(Mpf * L, 'econ');  % reactangular decomp.
  rep.sv = diag(Sl);
  T = Ul(:, 1:n) * diag(rep.sv(1:n));
  Ti = diag(1./rep.sv(1:n)) * Ul(:, 1:n)';
end

% Transform & truncate predictor to n states
A = Ti * Apf * T;
B = Ti * Bpf;
C = Cpf * T;
D = Dpf;

% Step 3: pull out the system (A,B,C,D) from the 
% reduced/stable predictor state space form.
% Return data in output struct rep.
rep.K = B(:, (nu+1):end);
rep.D = D(:, 1:nu) * (rmsy / rmsu);
rep.C = C;
rep.B = (B(:, 1:nu) + rep.K * D(:, 1:nu)) * (rmsy / rmsu);
rep.A = A + rep.K * rep.C;

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

function [Ghat, nt, ZZt, YZt] = batchdvarxestcov(DB, p, dterm, scl_yu)
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
  Zp = zeros(nzp + nu, Neff);
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
