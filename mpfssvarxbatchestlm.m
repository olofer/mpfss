function [r,DX1E]=mpfssvarxbatchestlm(DB,ords,dterm,varargin)
% function [r,DX1E]=mpfssvarxbatchestlm(DB,ords,dterm,varargin)
% Implementation/rehash of a predictor-form SSARX (Jansson 2003) subspace
% system identification method. This code handles D=0 and D!=0 as
% indicated by the flag "dterm" (see below). Uses CVA-weighted SVD.
%
% ords = [f p q n1 n2 dn] = [future past varx n1 n2 dn] sizes
% Uses VARX of order q>=f-1 for preestimation of the Markov blocks.
% Input-free (u=[]) stochastic SSARX not yet implemented.
% It must hold n1<=n2<=ny*f, optional stride in n is dn
%
% Returned system matrices are { A, B, C [,D], K [,L] }. 
% A-K*C and A+B*L are "supposed" to be stable matrices.
%
% Dataset DB must be a cell array of structures containing batches
% of input-output MIMO data DB{j}.{u,y} of compatible dimensions.
% Tries to use small memory footprint by squaring the least-squares
% data batch by batch.
%
% If called with ords=[b] then a single b-lag VARX model is returned.
% If numel(ords)==5 then a set of models is computed for 
% orders n1<=n<=n2.
%
% dterm=[d1 d2 d3] where d1 indicates if a direct term should be used, and
% d2 indicates if the residual covariance matrix is to be returned;
% d3 indicates whether a state-feedback injection estimate
% should be returned (only makes sense for closed-loop sysid.)
%
% Data is *assumed to be detrended* already. But scaling for conditioning
% is automatically done within this routine.
%
% Valid "varargin" options are:
%   'silent', 'no-cva', 'no-square', 'nok', 'noscale'
%
% It is also possible to pass a struct extra argument with pre-VARX
% Markov blocks (obtained with fancy regularisation perhaps);
% The return-structure from "varxgcv.m" may be used directly.
%
% Proper multibatch/block MCCV and .632 PE estimation for order
% selection can be done using the simp632cv12(..) function
% (which invokes this one).
%
% k erik j olofsson, april/may 2011, august 2011,
% june/july 2012, august/september/october 2012 updates


% TODO: general code cleanup + proper comments
% TODO: optional channel-by-channel diagonal signal scaling autodetect
% TODO: input-free implementation, ie. u=[]
% TODO: predictor-free output-error call variation (for open-loop systems)
%       (or rather assume the OE structure when solving for the matrices)
% TODO: improved pre-estimation by subsampling/regularisation ?


r=[];
DX1E=[];

dvarxcomptyp=0;     % batch-by-batch LS data squaring for ARX
abcdlstype=0;       % - " - for A,B,C,D
svdweight='cva';
regrtype='AkBk';
vrbsty=1;
no_k_matrix=0;
no_scale=0;
rq=[];

if numel(varargin)>=1
    for jj=1:numel(varargin)
        if ischar(varargin{jj})
            if strcmp(varargin{jj},'silent')        % non-verbose mode
                vrbsty=0;
            elseif strcmp(varargin{jj},'no-cva')    % turn-off CVA weight
                svdweight='none';
            elseif strcmp(varargin{jj},'no-square') % use non-squaring LS solution
                dvarxcomptyp=1;
                abcdlstype=1;
            elseif strcmp(varargin{jj},'nok')       % no K matrix, output-error structure assumed
                no_k_matrix=1;
            elseif strcmp(varargin{jj},'noscale')   % turn off auto-standardisation?
                no_scale=1;
            else
                error('unrecognised extra string option given.');
            end
        elseif isstruct(varargin{jj})
            if isfield(varargin{jj},'Ghat')
                % Assume here the dimensions are compatible with the "ords"
                % and "dterm" arguments, will be checked later.
                rq=varargin{jj};
            else
                error('unrecognised extra struct option given.');
            end
        else
            disp(['*** warning: extra argument #',num2str(jj),' unresolved.']);
        end
    end
end

vdisp(['mb/ssarx/lm invoked: svdweight="',...
    svdweight,'"; regrtype="',regrtype,...
    '"; lstypes=',num2str(dvarxcomptyp),',',num2str(abcdlstype)],vrbsty);

if nargin==2
    dterm=0;
end

if numel(dterm)==3
    lterm=dterm(3);
    reeterm=dterm(2);
    dterm=dterm(1);
elseif numel(dterm)==2
    lterm=0;
    reeterm=dterm(2);
    dterm=dterm(1);
elseif numel(dterm)==1
    lterm=0;
    reeterm=0;
    dterm=dterm(1);
else
    error('dterm=[d1 d2 d3] or dterm=[d1 d2] or dterm=[d1] required.');
end

if dterm
    vdisp('Assuming nonzero direct feedthrough (D!=0).',vrbsty);
    dterm=1;    % in case >1
else
    vdisp('Assuming zero direct feedthrough (D=0).',vrbsty);
end

if no_k_matrix
    vdisp('Assuming output-error structure (K=0).',vrbsty);
end

% construction of empty example DB={[]};
if ~iscell(DB)
    error('"DB" must be a cell array of batch data');
end

nb=numel(DB);

if nb==0
    error('no batches in dataset');
end

ny=size(DB{1}.y,1);
nu=size(DB{1}.u,1);

vdisp(['Expects #',num2str(ny),' outputs and #',num2str(nu),...
    ' inputs in #',num2str(nb),' batches.'],vrbsty);

% Compute RMS of the input-output dataset
% At the same time check that the matrices' sizes make sense.
nbb=zeros(nb,1);
Ruu=zeros(nu,nu);
Ryy=zeros(ny,ny);
mu=zeros(nu,1);
my=zeros(ny,1);
for bb=1:nb
    Nb1=size(DB{bb}.y,2); Nb2=size(DB{bb}.u,2);
    if Nb1~=Nb2
        error('Dataset length (N) mismatch.');
    end
    nbb(bb)=Nb1;    % record length of each batch #bb (used for preallocation below)
    ny1=size(DB{bb}.y,1); nu1=size(DB{bb}.u,1);
    if ny1~=ny || nu1~=nu
        error('Dataset input-output (nu,ny) size mismatch.');
    end
    if ny1>=Nb1 || nu1>=Nb1
        error('Output and/or input size larger than dataset length.');
    end
    Ruu=Ruu+DB{bb}.u*DB{bb}.u';
    Ryy=Ryy+DB{bb}.y*DB{bb}.y';
    mu=mu+sum(DB{bb}.u,2);
    my=my+sum(DB{bb}.y,2);
end
Nb=sum(nbb);
Ruu=Ruu/Nb;
Ryy=Ryy/Nb;
rmsu=sqrt(trace(Ruu)/nu);
rmsy=sqrt(trace(Ryy)/ny);
mu=mu/Nb;
my=my/Nb;

vdisp(['sum_b(N_b)=',num2str(Nb),' total vector (input,output) samples.'],vrbsty);

% TODO: issue warning if the individual channels have widely varying RMSs.
% TODO: remove possibly significant mean of vector ?

r.Ryy=Ryy;
r.Ruu=Ruu;
r.rmsyu=[rmsy,rmsu];
r.myu=[my;mu];

if ~no_scale
    % Rescale (u,y)-data to RMS unity
    for bb=1:nb
        DB{bb}.y=(1/rmsy)*DB{bb}.y;
        DB{bb}.u=(1/rmsu)*DB{bb}.u;
    end
else
    rmsy=1; rmsu=1; % no back-scaling to be applied below (!)
end

% TODO: issue warning if my and rmsy have almost the same size.

if isempty(ords)
    error('"ords" cannot be empty.');
end

if numel(ords)==1
    vdisp(['Computes a VARX model of lag order q=',num2str(ords(1)),'...'],vrbsty);
%     error('ARX multibatch k-fold CV not yet implemented.');
    rq=batchdvarxestcov(DB,ords(1),dterm,dvarxcomptyp); % simply return a (D)VARX
    r.Hhat=rq.Ghat;
    return;
end

if numel(ords)>=3
    tssarx=tic;
    f=ords(1);
    p=ords(2);
    q=ords(3);
    if q<f-1
        error('q>=f-1 required.');
    end
    setoforders=0;
    
    % ords=[f p q n] or ords=[f p q n1 n2] ??
    
    if numel(ords)==3
        n=ny*f;
    elseif numel(ords)==4
        n=ords(4);
        if n<=0 || n>ny*f
            n=ny*f; % default to untruncated if bogus input
        end
    elseif numel(ords)==5 || numel(ords)==6
        n1=ords(4);
        n2=ords(5);
        if n1>n2 || n2>ny*f
            error('n1<=n2<=ny*f required.');
        end
        n=n2;
        setoforders=1;
        if numel(ords)==6 && ords(6)>=1
            dn=ords(6);
        else
            dn=1;
        end
    else
        error('too many elements in "ords".');
    end
    
    SPP=Nb/(q*ny);   % approximate "samples per parameter" ratio
    
    vdisp(['Using f=',num2str(f),', p=',num2str(p),...
        ', q=',num2str(q),' and n=',num2str(n),' (f*n_y=',num2str(f*ny),...
        ', spp=',num2str(SPP),')'],vrbsty);
    
    % Multibatch SSARX rehash embark here
    fpqn=[f p q n];
    
    % 1. VARX preestimation of order q
    if isempty(rq)
        rq=batchdvarxestcov(DB,q,dterm,dvarxcomptyp);
    else
        % "rq" provided as extra argument; check dimensions and rescale to
        % standardised signal sizes.
        if size(rq.Ghat,1)~=ny || size(rq.Ghat,2)~=(q*ny+q*nu+dterm*nu)
            error('incompatible dimensions: given varx structure does not match q,dterm.');
        end
        
        vdisp('Using provided pre-varx Markov blocks.',vrbsty);
        rq.Ghat=rq.Ghat*(rmsu/rmsy);
    end
    
    % 2. Use the multibatch VARX coefficients to pretreat dataset for CVA
%     [Jn,sigm]=batchyfzp(DB,rq,fpqn,dterm,svdweight);
    [V,sigm,S]=batchyfzp(DB,rq,fpqn,dterm,svdweight);
    % NB: n is not utilised in the above function call.
    
    if setoforders
        vdisp(['Forming a sequence of truncated models ',num2str(n1),'<=n<=',num2str(n2)],vrbsty);
        if dn>1
            vdisp(['Order stride is dn=',num2str(dn)],vrbsty);
        end
        r.sigm=sigm;
        r.mm=(n1:dn:n2)';
        r.rmm={[]};
        nm=numel(r.mm);
        if nargout>1
            DX1E={[]};
            vdisp('Innovations and initial states requested.',vrbsty);
        end
        for jj=1:nm
            mm=r.mm(jj);
            vdisp(['#',num2str(jj),'/',num2str(nm),'; n=',num2str(mm)],vrbsty);
            Jmm=truncatesvd(V,S,mm,svdweight);
            [AK,BK,C,D]=batchabcdfromstate(DB,[f p q mm],Jmm,dterm,regrtype,abcdlstype);
            if nargout>1
                [Ree,DX1Ej]=batchresidualcov(DB,fpqn,AK,BK,C,D,rmsy);
                DX1E{jj}=DX1Ej;
            else
                if reeterm
                    Ree=batchresidualcov(DB,fpqn,AK,BK,C,D,rmsy);
                else
                    Ree=[];
                end
            end
            [A,B,K,C,D,Ree]=getinnov(AK,BK,C,D,Ree,nu,rmsy,rmsu);
            r.rmm{jj}.A=A; r.rmm{jj}.B=B; r.rmm{jj}.K=K;
            r.rmm{jj}.C=C; r.rmm{jj}.D=D; r.rmm{jj}.Ree=Ree;
        end
        tssarx=toc(tssarx);
        vdisp(['done (clocked at ',num2str(tssarx),' sec).'],vrbsty);
        return;
    end
    
    % Truncate the state sequence
    Jn=truncatesvd(V,S,n,svdweight);
    clear V; clear S;
    
    % 3. Use Jn to estimate a state sequence and then estimate the state-matrices by least-squares
    if no_k_matrix
        [AK,BK,C,D]=batchabcdfromstate_nok(DB,fpqn,Jn,dterm,regrtype,abcdlstype);
    else
        [AK,BK,C,D]=batchabcdfromstate(DB,fpqn,Jn,dterm,regrtype,abcdlstype);
    end
    
    if lterm
        vdisp('input-injection requested.',vrbsty);
        L=batchlfromstate(DB,fpqn,Jn);
    else
        L=[];
    end
    
    % 4. (mayhaps optional) For each batch individually, estimate initial state and innovations
    if nargout>1
        vdisp('Innovations and initial states requested.',vrbsty);
        % Loop through the batches and evaluate x(1) and {e(k)}_k given (A,B,C,D,K)
        [Ree,DX1E]=batchresidualcov(DB,fpqn,AK,BK,C,D,rmsy);
    else
        if reeterm
            vdisp('residual covariance evaluation requested.',vrbsty);
            Ree=batchresidualcov(DB,fpqn,AK,BK,C,D,rmsy);
        else
            Ree=[];
        end
    end
    
    % 5. Return model (rescaled for original data)
    [A,B,K,C,D,Ree]=getinnov(AK,BK,C,D,Ree,nu,rmsy,rmsu);
    r.A=A; r.B=B; r.K=K;
    r.C=C; r.D=D; r.Ree=Ree;
    r.fpqn=fpqn;
    r.sigm=sigm;
    r.spp=SPP;
    r.L=L*(rmsu/rmsy);

    tssarx=toc(tssarx);
    vdisp(['done (clocked at ',num2str(tssarx),' sec).'],vrbsty);
else
    error('need to provide ords=[f p q] at least.');
end

end

%% Auxiliary subspace identification functions

function [A,B,K,C,D,Ree]=getinnov(AK,BK,C,D,R,nu,rmsy,rmsu)
if isempty(D)
    K=BK(:,(nu+1):end);
    A=AK+K*C;
    B=BK(:,1:nu)*(rmsy/rmsu);
    C=C;
    D=[];
else
    K=BK(:,(nu+1):end);
    A=AK+K*C;
    B=(BK(:,1:nu)+K*D)*(rmsy/rmsu);
    C=C;
    D=D*(rmsy/rmsu);
end
Ree=R*rmsy*rmsy;
end

function [V,sigm,S]=batchyfzp(DB,rq,fpqn,dterm,svdweight)
% function [Jn,sigm,S]==batchyfzp(DB,rq,fpqn,dterm,svdweight)
% Given the VARX preestimate matrix coefficients in structure rq
% and the batched dataset DB, form the covariance matrices needed
% for canonical variate analysis and return singular values and the 
% nontruncated state projection matrix V
Hq=rq.Ghat;
f=fpqn(1); p=fpqn(2); q=fpqn(3); n=fpqn(4);
if dterm==1
    [Gfbarb,Dfb]=GDfromH(Hq,f,q,size(DB{1}.u,1));
else
    Gfbarb=GDfromH(Hq,f,q,0);
end
for bb=1:numel(DB)
    [Yfb,Ufb,Zpb,Zf1b]=pbsiddata(DB{bb}.y,DB{bb}.u,[f p]);
    if dterm==1
        Yftb=Yfb-Gfbarb*Zf1b-Dfb*Ufb;
    else
        Yftb=Yfb-Gfbarb*Zf1b;
    end
    if bb==1
        YftYft=Yftb*Yftb';
        YftZp=Yftb*Zpb';
        ZpZp=Zpb*Zpb';
    else
        YftYft=YftYft+Yftb*Yftb';
        YftZp=YftZp+Yftb*Zpb';
        ZpZp=ZpZp+Zpb*Zpb';
    end
end
% Now do CVA weighted SVD and truncation to n
if strcmp(svdweight,'none')==1
    % Least-squares estimation of the H_{fp} Hankel matrix
    Hfp=YftZp/ZpZp;
    [U,S,V]=svd(Hfp,'econ');
    sigm=diag(S);
    S=sigm;
%     V=V(:,1:n);
%     Lp=diag(sqrt(sigm(1:n)))*V';
elseif strcmp(svdweight,'cva')==1
    % canonical variate analysis approach to get a proper "Lp"
    Syy=YftYft;
    Szz=ZpZp;
    [U,S,V]=svd(Syy); SRy=U*diag(1./sqrt(diag(S)))*V';
    [U,S,V]=svd(Szz); SRz=U*diag(1./sqrt(diag(S)))*V';
    M=SRy*(YftZp)*SRz;
    [U,S,V]=svd(M,'econ');
    sigm=diag(S);
    S=SRz;
%     V=V(:,1:n);
%     Lp=V'*SRz;
else
    error('unrecognised SVD weighting.');
end
end

function Jn=truncatesvd(V,S,n,svdweight)
if strcmp(svdweight,'none')==1
    V=V(:,1:n);
    Jn=diag(sqrt(S(1:n)))*V';
elseif strcmp(svdweight,'cva')==1
    V=V(:,1:n);
    Jn=V'*S;
else
    error('unrecognised SVD weighting.');
end
end

function [G,Df]=GDfromH(H,f,q,nu)
% Assemble the Gfbar matrix from the VARX preestimated
% coefficients in H=[H(1)..H(q) D]
% Assumes H=[H(1)..H(q)] if nu==0
% TODO: recheck validity
ny=size(H,1); nz=(size(H,2)-nu)/q;
G=zeros(f*ny,(f-1)*nz);
for rr=1:f
    for cc=1:(f-1)
        if rr>cc
            qq=rr-cc;
            G((1+(rr-1)*ny):(rr*ny),(1+(cc-1)*nz):(cc*nz))=...
                H(:,(1+(qq-1)*nz):(qq*nz));
        end
    end
end
if nargout>1
    if nu==0
        error('extra output (Df) makes no sense.');
    end
    % only makes sense if nu>0
    D=H(:,(nz*q+1):end);
    Df=kron(eye(f),D);
end
end

function [Yf,Uf,Zp,Zf1]=pbsiddata(Y,U,ords)
% Assemble the data matrices Y_f, Z_p and Z_{f-1} for a single batch
ny=size(Y,1); nu=size(U,1);
N=size(Y,2); f=ords(1); p=ords(2);  % ords=[f p]
k1=1+p; k2=N+1-f;
Neff=k2-k1+1;
Z=[U;Y];        % not very efficient to actually form this one; optimise this.
Yf=zeros(ny*f,Neff);
Uf=zeros(nu*f,Neff);
Zp=zeros((nu+ny)*p,Neff);
Zf1=zeros((nu+ny)*(f-1),Neff);
for k=k1:k2
    kk=k-k1+1;
    Yf(:,kk)=reshape(Y(:,k:1:(k+f-1)),size(Yf,1),1);
    Uf(:,kk)=reshape(U(:,k:1:(k+f-1)),size(Uf,1),1);
    Zp(:,kk)=reshape(Z(:,(k-1):-1:(k-p)),size(Zp,1),1);
    Zf1(:,kk)=reshape(Z(:,k:1:(k+f-2)),size(Zf1,1),1);
end
end

function L=batchlfromstate(DB,fpqn,Lp)
% Least-squares estimation of the input-injection state-feedback L
% in the sense u(k)=L*x(k)+delta, with x(k) the state as defined by the
% projection Lp.
p=fpqn(2); % n=fpqn(4);
% LS data squaring for larger data sets
Y=DB{1}.y; U=DB{1}.u;
[Xk,k1,k2]=XfromLp(Y,U,Lp,p);
% NX=size(Xk,2);
Un=Y(:,k1:k2); Xn=[Xk;];
UnXnt=Un*Xn'; XnXnt=Xn*Xn';
for bb=2:numel(DB)
    Y=DB{bb}.y; U=DB{bb}.u;
    [Xk,k1,k2]=XfromLp(Y,U,Lp,p); % NX=size(Xk,2);
    Un=Y(:,k1:k2); Xn=[Xk;];
    UnXnt=UnXnt+Un*Xn'; XnXnt=XnXnt+Xn*Xn';
end
L=UnXnt/XnXnt; % LS estimate of L (minimises the Frobenius norm of "delta")
end

function [AK,BK,C,D]=batchabcdfromstate(DB,fpqn,Lp,dterm,regrtype,lstype)
% Note: does not return residuals, since these are not "true" residuals:
% ie. the innovations need to be calculated differently, by recursion of
% the KF, batch by batch, from estimated initial conditions
p=fpqn(2); n=fpqn(4);
if nargout~=4
    error('suspicious function call.');
end
if dterm
    if strcmp(regrtype,'AkBk')==1
        if lstype==0
            % More appropriate for large data (N) situations (?!)
            Y=DB{1}.y; U=DB{1}.u;
            [Xk,k1,k2]=XfromLp(Y,U,Lp,p);
            NX=size(Xk,2);
            Yn=Y(:,k1:k2); Xn=[Xk;U(:,k1:k2)];
            YnXnt=Yn*Xn'; XnXnt=Xn*Xn';
            Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1));Y(:,k1:(k2-1))];
            Xn1Xnzt=Xn1*Xnz'; XnzXnzt=Xnz*Xnz';
            for bb=2:numel(DB)
                Y=DB{bb}.y; U=DB{bb}.u;
                [Xk,k1,k2]=XfromLp(Y,U,Lp,p); NX=size(Xk,2);
                Yn=Y(:,k1:k2); Xn=[Xk;U(:,k1:k2)];
                YnXnt=YnXnt+Yn*Xn'; XnXnt=XnXnt+Xn*Xn';
                Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1));Y(:,k1:(k2-1))];
                Xn1Xnzt=Xn1Xnzt+Xn1*Xnz'; XnzXnzt=XnzXnzt+Xnz*Xnz';
            end
            AB=Xn1Xnzt/XnzXnzt; % AB=[A_K B_K]
            CD=YnXnt/XnXnt;     % CD=[C D]
            AK=AB(:,1:n);
            BK=AB(:,(n+1):end);
            C=CD(:,1:n);
            D=CD(:,(n+1):end);
        elseif lstype==1
            % Uses growing arrays, but does not square data for LS (?!)
            Yn=[]; Xn=[]; Xn1=[]; Xnz=[];
            for bb=1:numel(DB)
                Y=DB{bb}.y; U=DB{bb}.u;
                [Xk,k1,k2]=XfromLp(Y,U,Lp,p);
                NX=size(Xk,2);
                Xn1=[Xn1,Xk(:,2:NX)];
                Xnz=[Xnz,[Xk(:,1:(NX-1));U(:,k1:(k2-1));Y(:,k1:(k2-1))]];
                Yn=[Yn,Y(:,k1:k2)];
                Xn=[Xn,[Xk;U(:,k1:k2)]];
            end
            % Ak,Bk and C and D indirectly via state sequence estimation
            AB=Xn1/Xnz; % AB=[A_K B_K]
            CD=Yn/Xn;   % CD=[C D]
            AK=AB(:,1:n);
            BK=AB(:,(n+1):end);
            C=CD(:,1:n);
            D=CD(:,(n+1):end);
        else
            error('unrecognised LS-type request.');
        end
    elseif strcmp(regrtype,'ABK')==1
        error('not implemented');
    else
        error('unrecognised regression type');
    end
else
    % D=0 assumed
    if strcmp(regrtype,'AkBk')==1
        if lstype==0
            % LS data squaring for larger data sets
            Y=DB{1}.y; U=DB{1}.u;
            [Xk,k1,k2]=XfromLp(Y,U,Lp,p);
            NX=size(Xk,2);
            Yn=Y(:,k1:k2); Xn=[Xk;];
            YnXnt=Yn*Xn'; XnXnt=Xn*Xn';
            Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1));Y(:,k1:(k2-1))];
            Xn1Xnzt=Xn1*Xnz'; XnzXnzt=Xnz*Xnz';
            for bb=2:numel(DB)
                Y=DB{bb}.y; U=DB{bb}.u;
                [Xk,k1,k2]=XfromLp(Y,U,Lp,p); NX=size(Xk,2);
                Yn=Y(:,k1:k2); Xn=[Xk;];
                YnXnt=YnXnt+Yn*Xn'; XnXnt=XnXnt+Xn*Xn';
                Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1));Y(:,k1:(k2-1))];
                Xn1Xnzt=Xn1Xnzt+Xn1*Xnz'; XnzXnzt=XnzXnzt+Xnz*Xnz';
            end
            AB=Xn1Xnzt/XnzXnzt; % AB=[A_K B_K]
            C=YnXnt/XnXnt;
            AK=AB(:,1:n);
            BK=AB(:,(n+1):end);
            D=[];
        elseif lstype==1
            % data accumulation ok for smaller data sets
            Yn=[]; Xn=[];
            Xn1=[]; Xnz=[];
            for bb=1:numel(DB)
                Y=DB{bb}.y; U=DB{bb}.u;
                [Xk,k1,k2]=XfromLp(Y,U,Lp,p);
                NX=size(Xk,2);
                Xn1=[Xn1,Xk(:,2:NX)];
                Xnz=[Xnz,[Xk(:,1:(NX-1));U(:,k1:(k2-1));Y(:,k1:(k2-1))]];
                Yn=[Yn,Y(:,k1:k2)];
                Xn=[Xn,Xk];
            end
            AB=Xn1/Xnz; % AB=[A_K B_K]
            AK=AB(:,1:n);
            BK=AB(:,(n+1):end);
            C=Yn/Xn;   % C=[C]
            D=[];
        else
            error('unrecognised LS-type request.');
        end
    elseif strcmp(regrtype,'ABK')==1
        error('not implemented');
    else
        error('unrecognised regression type');
    end
end

end

function [AK,BK,C,D]=batchabcdfromstate_nok(DB,fpqn,Lp,dterm,regrtype,lstype)
% Same as "batchabcdfromstate" but with K=0 prescribed
p=fpqn(2); n=fpqn(4);
if nargout~=4
    error('suspicious function call.');
end
if dterm
    % D=arbitrary, K=0
    if strcmp(regrtype,'AkBk')==1
        if lstype==0
            % More appropriate for large data (N) situations (?!)
            Y=DB{1}.y; U=DB{1}.u; ny=size(Y,1);
            [Xk,k1,k2]=XfromLp(Y,U,Lp,p);
            NX=size(Xk,2);
            Yn=Y(:,k1:k2); Xn=[Xk;U(:,k1:k2)];
            YnXnt=Yn*Xn'; XnXnt=Xn*Xn';
            Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1))];
            Xn1Xnzt=Xn1*Xnz'; XnzXnzt=Xnz*Xnz';
            for bb=2:numel(DB)
                Y=DB{bb}.y; U=DB{bb}.u;
                [Xk,k1,k2]=XfromLp(Y,U,Lp,p); NX=size(Xk,2);
                Yn=Y(:,k1:k2); Xn=[Xk;U(:,k1:k2)];
                YnXnt=YnXnt+Yn*Xn'; XnXnt=XnXnt+Xn*Xn';
                Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1))];
                Xn1Xnzt=Xn1Xnzt+Xn1*Xnz'; XnzXnzt=XnzXnzt+Xnz*Xnz';
            end
            AB=Xn1Xnzt/XnzXnzt; % AB=[A B]
            CD=YnXnt/XnXnt;     % CD=[C D]
            AK=AB(:,1:n);
            BK=[AB(:,(n+1):end),zeros(n,ny)];   % pad with zeros for K
            C=CD(:,1:n);
            D=CD(:,(n+1):end);
        else
            error('lstype!=0 for K=0 not implemented.');
        end
    else
        error('option N/A');
    end
else
    % D=0, K=0
    if strcmp(regrtype,'AkBk')==1
        if lstype==0
            % LS data squaring for larger data sets
            Y=DB{1}.y; U=DB{1}.u; ny=size(Y,1);
            [Xk,k1,k2]=XfromLp(Y,U,Lp,p);
            NX=size(Xk,2);
            Yn=Y(:,k1:k2); Xn=[Xk;];
            YnXnt=Yn*Xn'; XnXnt=Xn*Xn';
            Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1))];
            Xn1Xnzt=Xn1*Xnz'; XnzXnzt=Xnz*Xnz';
            for bb=2:numel(DB)
                Y=DB{bb}.y; U=DB{bb}.u;
                [Xk,k1,k2]=XfromLp(Y,U,Lp,p); NX=size(Xk,2);
                Yn=Y(:,k1:k2); Xn=[Xk;];
                YnXnt=YnXnt+Yn*Xn'; XnXnt=XnXnt+Xn*Xn';
                Xn1=Xk(:,2:NX); Xnz=[Xk(:,1:(NX-1));U(:,k1:(k2-1))];
                Xn1Xnzt=Xn1Xnzt+Xn1*Xnz'; XnzXnzt=XnzXnzt+Xnz*Xnz';
            end
            AB=Xn1Xnzt/XnzXnzt; % AB=[A B]
            C=YnXnt/XnXnt;
            AK=AB(:,1:n);
            BK=[AB(:,(n+1):end),zeros(n,ny)];   % pad with zeros for K
            D=[];
        else
            error('lstype!=0 for K=0 not implemented.');
        end
    else
        error('option N/A');
    end
end
end

function [Xk,k1,k2]=XfromLp(Y,U,Lp,p)
% Estimate the state sequence given the projection matrix Lp
ny=size(Y,1); nu=size(U,1);
N=size(Y,2); k1=1+p; k2=N;
Neff=k2-k1+1; nx=size(Lp,1);
Z=[U;Y];
Xk=zeros(nx,Neff);
for k=k1:k2
    kk=k-k1+1;
    Xk(:,kk)=Lp*reshape(Z(:,(k-1):-1:(k-p)),p*(ny+nu),1);
end
end

%% Methods for innovations and initial state

function [REE,DX1E]=batchresidualcov(DB,fpqn,AK,BK,C,D,rmsy)
% Loop through each batch of data and estimate innovations and
% the initial states.
if nargout>1
    DX1E={[]};
end
f=fpqn(1); p=fpqn(2); q=fpqn(3);
Hq=HfromABCD(AK,BK,C,D,q);
ny=size(DB{1}.y,1);
REE=zeros(ny,ny);
NEE=0;
for bb=1:numel(DB)
    Y=DB{bb}.y;
    U=DB{bb}.u;
    [x1,e1f]=initialest(Y,U,AK,C,Hq,q,p);
    Gp=onesteppfilterk(AK,BK,C,D);
    Yp=lsim(Gp,[U;Y]',[],x1);
    e=Y-Yp';
    Ne=size(e,2);
    Ree=e*e';
    REE=REE+Ree;
    NEE=NEE+Ne;
    if nargout>1
        % return scaled-back data
%        DX1E{bb}.e=e*rmsy;
%         DX1E{bb}.rho=rho;
        Ree=Ree/Ne;
        DX1E{bb}.Ree=Ree*rmsy*rmsy;
        DX1E{bb}.x1=x1*rmsy;
    end
end
REE=REE/NEE;
end

% function Gp=onesteppfilter(A,B,K,C,D)
% % Returns the predictor filter for the signal model A,B,K,C,D
% ny=size(C,1);
% % nu=size(B,2);
% if isempty(D) || (numel(D)==1 && D==0)
%     Gp=ss(A-K*C,[B,K],C,0,-1);
% else
%     Gp=ss(A-K*C,[B-K*D,K],C,[D,zeros(ny)],-1);
% end
% end

function Gp=onesteppfilterk(AK,BK,C,D)
ny=size(C,1);
% nu=size(B,2);
if isempty(D)
    Gp=ss(AK,BK,C,0,-1);
else
    Gp=ss(AK,BK,C,[D,zeros(ny)],-1);
end
end

% function DX1E=batchinitialest(DB,Lp,fpqn,AK,BK,C,D,rmsy)
% % Loop through each batch of data and estimate innovations and
% % the initial states.
% DX1E={[]};
% f=fpqn(1); p=fpqn(2); q=fpqn(3);
% Hq=HfromABCD(AK,BK,C,D,q);
% ny=size(DB{1}.y,1);
% for bb=1:numel(DB)
%     Y=DB{bb}.y;
%     U=DB{bb}.u;
%     [Xk,k1,k2]=XfromLp(Y,U,Lp,p);
%     if isempty(D)
%         e=Y(:,k1:k2)-C*Xk;
%     else
%         e=Y(:,k1:k2)-C*Xk-D*U(:,k1:k2);     % k=p+1..N
%     end
%     Ree=e*e';
%     rho=sqrt(trace(Ree))/sqrt(trace(Y(:,k1:k2)*Y(:,k1:k2)'));
%     Ree=Ree/(k2-k1+1);
%     % Estimate initial state and initial innovations k=1..p
%     [x1,e1f]=initialest(Y,U,AK,C,Hq,q,p);
%     e=[reshape(e1f,ny,p),e];    % append innovations k=1..N
%     
%     % return scaled-back data
%     DX1E{bb}.e=e*rmsy;
%     DX1E{bb}.rho=rho;
%     DX1E{bb}.Ree=Ree*rmsy*rmsy;
%     DX1E{bb}.x1=x1*rmsy;
% end
% end

function H=HfromABCD(AK,BK,C,D,q)
% Assemble H=[H(1)..H(q) D] from (A,B,C,D,q); Markov coefficients implied from
% state space system - to be fed back to the preestimation step.
% Put D=[] if no direct term should be present in H 
ny=size(C,1); nu=size(D,2); nz=size(BK,2); n=size(AK,1);
P=eye(n);
if isempty(D)
    H=zeros(ny,q*nz);
else
    H=zeros(ny,q*nz+nu);
end
for pp=1:q
    H(:,(1+(pp-1)*nz):(pp*nz))=C*P*BK;
    if pp<q
        P=P*AK;
    end
end
if ~isempty(D)
    H(:,(q*nz+1):end)=D;
end
end

function [x1,e1f]=initialest(Y,U,AK,C,H,q,f)
% Initial state estimation
nu=size(U,1); n=length(AK); ny=size(Y,1);
Gmp=zeros(ny*f,n);
P=eye(n);
for pp=1:f
    Gmp((1+(pp-1)*ny):(pp*ny),:)=C*P;   % observability matrix
    if pp<f
        P=P*AK;
    end
end
yf=reshape(Y(:,1:1:(1+f-1)),f*ny,1);
Z=[U(:,1:f);Y(:,1:f)];
zf1=reshape(Z(:,1:1:(1+f-2)),(f-1)*(nu+ny),1);
if size(H,2)==(q*(ny+nu)+nu)
    [G,Df]=GDfromH(H,f,q,nu);
    uf=reshape(U(:,1:1:(1+f-1)),f*nu,1);
    yft=yf-G*zf1-Df*uf;
else
    G=GDfromH(H,f,q,0);
    yft=yf-G*zf1;
end
x1=Gmp\yft;
e1f=yft-Gmp*x1;
end

%% Below is VARX-only routines including cross-validation/jackknifing

function r=batchdvarxestcov(DB,q,dterm,comptyp)
% function r=batchdvarxestcov(DB,q,dterm,comptyp)
% General estimation of vector autoregressive models (VARXs)
% with optional direct term D from input-output data;
% ASSUMES: detrending & signal scaling already done properly.
% VARX order is q (=na=nb).
% comptyp=0, batch-by-batch data squaring, =1 data array accumulation. 

% Standardised least-squares estimation; Y=G*Z
r=[];
nb=numel(DB);
if comptyp==0
    [Yb,Zb]=dvarxdata(DB{1}.y,DB{1}.u,q,dterm);
    YZt=Yb*Zb';
    ZZt=Zb*Zb';
    for bb=2:nb
        [Yb,Zb]=dvarxdata(DB{bb}.y,DB{bb}.u,q,dterm);
        YZt=YZt+Yb*Zb';
        ZZt=ZZt+Zb*Zb';
    end
    r.Ghat=YZt/ZZt;
elseif comptyp==1
    Y=[]; Z=[];
    for bb=1:nb
        [Yb,Zb]=dvarxdata(DB{bb}.y,DB{bb}.u,q,dterm);
        Y=[Y,Yb]; Z=[Z,Zb];
    end
    r.Ghat=Y/Z;
else
    error('bad "comptyp" given to DVARX routine.');
end
end

% Auxillary functions
function [Y,Zq]=dvarxdata(y,u,q,dterm)
ny=size(y,1); nu=size(u,1); N=size(y,2);
k1=q+1; k2=N; Neff=k2-k1+1; nz=ny+nu;
Z=[u;y];
Y=zeros(ny,Neff);
nzq=nz*q;
if dterm>0
    % augment with direct term
    Zq=zeros(nzq+nu,Neff);
    for k=k1:k2
        kk=k-k1+1;
        Y(:,kk)=y(:,k);
        Zq(:,kk)=[reshape(Z(:,(k-1):-1:(k-q)),nzq,1);u(:,k)];
    end
else
    % no direct term
    Zq=zeros(nzq,Neff);
    for k=k1:k2
        kk=k-k1+1;
        Y(:,kk)=y(:,k);
        Zq(:,kk)=reshape(Z(:,(k-1):-1:(k-q)),nzq,1);
    end
end
end

%% verbosity control
function vdisp(s,v)
if v>=1
    disp(s);
end
% possible todo: levels of verbosity control
end

