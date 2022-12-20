%[1995]-"Particle Swarm Optimization" 
%[1998]-"A modified particle swarm optimizer"

% (9/12/2020)

function PSO = jParticleSwarmOptimization(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.6;
c1    = 2;              % cognitive factor
c2    = 2;              % social factor 
w     = 0.9;            % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity 

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'c1'), c1 = opts.c1; end 
if isfield(opts,'c2'), c2 = opts.c2; end 
if isfield(opts,'w'), w = opts.w; end 
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Feature weight matrix
weight = zeros(1, dim);
% Initial 
X   = zeros(N,dim); 
V   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  
% Fitness
fit  = zeros(1,N); 
fitG = inf;
for i = 1:N 
  fit(i) = fun(feat,label,(X(i,:) > thres),opts); 
  % Gbest update
  if fit(i) < fitG
    Xgb  = X(i,:); 
    fitG = fit(i);
  end
end
% PBest
Xpb  = X; 
fitP = fit;
% Pre
curve = zeros(1,floor(max_Iter / 10));
fnum = zeros(1,floor(max_Iter / 10));
curve(1) = fitG;
fnum(1) = length(find((Xgb > thres) == 1));
t = 2;  
k = 1;
% Iterations
while t <= max_Iter
  for i = 1:N
    for d = 1:dim
      r1 = rand();
      r2 = rand();
      % Velocity update (2a)
      VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + ...
        c2 * r2 * (Xgb(d) - X(i,d));
      % Velocity limit
      VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
      V(i,d) = VB;
      % Position update (2b)
      X(i,d) = X(i,d) + V(i,d);
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Pbest update %% Here
    % Feature vector (new)
    fv_n = X(i,:) > thres;
    % Feature vector (old)
    fv_o = Xpb(i,:) > thres;
    if fit(i) < fitP(i)      
      % Changes in features (1:new features emerge, -1:old features
      % disappear)
      change = fv_n - fv_o;
      case_emerge = find(change(1,:) == 1);
      case_disapper = find(change(1,:) == -1);
      % Calculate the weight of those features
      for j = 1 : length(case_emerge)
          weight(1, case_emerge(j)) = weight(1, case_emerge(j)) + 1;
      end
      for j = 1 : length(case_disapper)
          weight(1, case_disapper(j)) = weight(1, case_disapper(j)) - 1;
      end
      Xpb(i,:) = X(i,:); 
      fitP(i)  = fit(i);
    else
      % Changes in features (1:old features, -1:new features)
      change = fv_o - fv_n;
      case_emerge = find(change(1,:) == 1);
      case_disapper = find(change(1,:) == -1);
      % Calculate the weight of those features
      for j = 1 : length(case_emerge)
          weight(1, case_emerge(j)) = weight(1, case_emerge(j)) + 1;
      end
      for j = 1 : length(case_disapper)
          weight(1, case_disapper(j)) = weight(1, case_disapper(j)) - 1;
      end
    end
    % Gbest update
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
%   if mod(t, 10) == 0
%      k = k + 1;
     curve(t) = fitG; 
     fnum(t) = length(find((Xgb > thres) == 1));
%   end
  fprintf('\nIteration %d Best (PSO)= %f',t,fitG)
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
PSO.sf = Sf; 
PSO.ff = sFeat;
PSO.nf = length(Sf);
PSO.c  = curve;
PSO.fnum = fnum;
PSO.f  = feat;
PSO.l  = label;
PSO.fitG = fitG;
PSO.w = weight;
PSO.bi = Xgb;
end



