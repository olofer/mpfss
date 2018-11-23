%
% Illustration of the regularization auto-tune feature 
% implemented in mpfvarx(.).
%
% Use a long VARX model; fit it with regularization; then extract model
% with weighted model reduction based on the regularized markov parameters
%
% It is unclear how useful this is though.
%

if exist('dat', 'var') == 0
  dat = test_mpfss(10, 500); % make test data
else
  assert(iscell(dat));
  fprintf(1, 'recycling workspace dat\n');
end
p = 300;
n = 10;  % true system has three states
dterm = 0;
autoscl = 1;
cholla = -1;
mpfvarx(dat, [p, n], dterm, autoscl, cholla, 2.^(-8:8)); % make a plot using lambda vector and LOBO CV
rep0 = ans;
fprintf(1, 'selecting lambda = %e\n', rep0.lambda_select_single);
rep1 = mpfvarx(dat, [p, n], dterm, autoscl, cholla, 0.0);
rep2 = mpfvarx(dat, [p, n], dterm, autoscl, cholla, rep0.lambda_select_single);

figure;
sigma(ss(rep1.A, rep1.B, rep1.C, 0, -1), 'b-', ss(rep2.A, rep2.B, rep2.C, 0, -1), 'r-');
legend('long VARX no regul.', 'long VARX with regul.');
title('Open-loop system estimated SVs');
