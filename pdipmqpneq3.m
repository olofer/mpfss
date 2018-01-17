function rep = pdipmqpneq3(H, h, E, f, kmax, epstop, eta)
% function rep = pdipmqpneq3(H, h, E, f, kmax, epstop, eta)
% Solves the quadratic program
%
%   min_z (0.5 * z'*H*z + h'*z), s.t. E*z <= f     (QP)
%
% using a primal-dual interior point method based on 
% the "standard" Mehrotra predictor-corrector method.
% The initial point is automatically set.
% Uses a relative norm stop condition.
%
% Run test_pdipmqpneq3() for a health check
% (requires a reference QP solver).

rep = struct;
rep.cholerror = 0;
rep.isconverged = false;

% Set some defaults if needed
if nargin < 7, eta = 0.95; end
if nargin < 6, epstop = 1e-6; end
if nargin < 5, kmax = 40; end

assert(numel(eta) == 1 && eta > 0 && eta < 1);
assert(numel(kmax) == 1);
assert(kmax >= 1 && kmax <= 250);

% Dim. consistency checking
nx = size(H, 1); % # primal variables
assert(size(H, 2) == nx);
assert(size(h, 1) == nx);
assert(size(h, 2) == 1);
nz = size(E, 1); % # inequality constraints
assert(size(E, 2) == nx);
assert(size(f, 1) == nz);
assert(size(f, 2) == 1);

if numel(epstop) == 1
  epstop = ones(3, 1) * epstop;
end

assert(numel(epstop) == 3);
assert(min(epstop) > 0 && max(epstop) <= 1e-2);

normhinf = norm(h, 'inf');
normfinf = norm(f, 'inf');
etainf = max([normhinf; normfinf; norm(E(:), 'inf'); norm(H(:), 'inf')]);

thrL = (normhinf + 1) * epstop(1);
thrs = (normfinf + 1) * epstop(2);
thrmu = epstop(3);

% Initial point
x = zeros(nx, 1);
z = ones(nz, 1) * sqrt(etainf);
s = ones(nz, 1) * sqrt(etainf);
e = ones(nz, 1);

k = 0;
rL = H*x+h+E'*z;
rs = s+E*x-f;
rsz = s.*z;
mu = sum(z.*s)/nz;
while (k < kmax && (norm(rL, 'inf') >= thrL || norm(rs, 'inf') >= thrs || mu >= thrmu))
  r_bar = -E'*((rsz-z.*rs)./s);
  h_bar = -(rL+r_bar);
  tmp1 = diag(sqrt(z./s))*E;
  H_bar = H+tmp1'*tmp1;
  [L, pp] = chol(H_bar, 'lower');
  if pp > 0
    rep.cholerror = pp;
    break;
  end
  dx_ = L'\(L\h_bar);
  ds_ = -rs-E*dx_;
  dz_ = -(rsz+z.*ds_)./s;
  akp = 1/max([1, max(-dz_./z), max(-ds_./s)]);
  sigma = (((z+akp*dz_)'*(s+akp*ds_))/(nz*mu))^3;
  rsz = rsz+dz_.*ds_-sigma*mu*e;
  r_bar = -E'*((rsz-z.*rs)./s);
  h_bar = -(rL+r_bar);
  dx = L'\(L\h_bar);
  ds = -rs-E*dx;
  dz = -(rsz+z.*ds)./s;
  akpc = 1; idxz = find(dz < 0); idxs = find(ds < 0);
  if numel(idxz) > 0
    akpc = min(akpc, 1/max(-dz(idxz)./z(idxz)));
  end
  if numel(idxs) > 0
    akpc = min(akpc, 1/max(-ds(idxs)./s(idxs)));
  end
  akpc = eta*akpc; 
  x = x+akpc*dx;
  z = z+akpc*dz;
  s = s+akpc*ds;
  k = k+1;
  rL = H*x+h+E'*z;
  rs = s+E*x-f;
  rsz = s.*z;
  mu = sum(rsz)/nz;
  assert(all(rsz > 0));
end

% Populate output struct 
rep.x = x;
rep.fx = 0.5*x'*H*x + h'*x;
rep.iters = k;
rep.epstop = epstop;
rep.maxiters = kmax;
rep.z = z;
rep.s = s;
rep.mu = mu;  % complementarity
rep.gam = norm(rsz, 'inf');  % centrality
rep.relinf_rL = norm(rL, 'inf') / normhinf;
rep.relinf_rs = norm(rs, 'inf') / normfinf;
rep.isconverged = (rep.relinf_rL < epstop(1) && rep.relinf_rs < epstop(2) && mu < epstop(3));
rep.solver = mfilename();

end
