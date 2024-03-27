%[1995]-"Particle Swarm Optimization" 
%[1998]-"A modified particle swarm optimizer"

% (14/12/2022)
% Developed by Xubin Wang, Email: wangxb19@mails.jlu.edu.cn

function PSO = jMultiTaskPSO(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;
c1    = 2;              % cognitive factor
c2    = 2;              % social factor 
c3    = 2;              % group social factor 
w     = 0.9;            % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity 

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'c1'), c1 = opts.c1; end 
if isfield(opts,'c2'), c2 = opts.c2; end
if isfield(opts,'c3'), c3 = opts.c3; end 
if isfield(opts,'w'), w = opts.w; end 
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
% dim = size(features_after_reduction,2); 
dim = size(feat, 2);
% Feature weight matrix
weight = zeros(1, dim);
% Initial 
X   = zeros(N,dim); 
V   = zeros(N,dim); 
for i = 1:N
   for d = 1:dim
       X(i,d) =  lb + (ub - lb) * rand();
   end
end  
% Fitness
fit  = zeros(1,N); 
fitG = inf;
% Number of subpopulation
numSub = 2;
fitSub = ones(1,numSub)*inf;
Xsub = zeros(numSub,dim);

subSize = floor(N/numSub);
j = 1;
for i = 1:N
%   Z(features_after_reduction(X(i,:) > thres)) = 1;  
  fit(i) = fun(feat,label,(X(i,:) > thres),opts); 
  % SubBest update
  if fit(i) < fitSub(j)
      Xsub(j,:) = X(i,:);
      fitSub(j) = fit(i);
  end
  % Subpopulation update
  if mod(i,subSize) == 0     
      j = j + 1;
  end
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
curve = zeros(1,max_Iter);
curve(1) = fitG;
fnum = zeros(1,max_Iter);
fnum(1) = length(find((Xpb > thres) == 1));
t = 2;  
% Iterations
while t <= max_Iter
  k = 1; 
  for i = 1:N
    if k == 1 % Subpopulation 1
        for d = 1:dim
          r1 = rand();
          r2 = rand();
          r3 = rand();
          % Velocity update (2a)
          VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + ...
            c2 * r2 * (Xgb(d) - X(i,d)) + c3 * r3 * (Xsub(2,d) - X(i,d));
          % Velocity limit
          VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
          V(i,d) = VB;
          X(i,d) = X(i,d) + V(i,d);
        end
        % Boundary
        XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
        X(i,:) = XB;
    else % Subpopulation 2
        index = weight < 0;
        valued_features = weight;
        valued_features(index) = 0; % Let the features with value less than 0 equal to 0
        sum_values = sum(valued_features);
        for d = 1 : dim
            p = rand(); % Probabilty
            if (valued_features(d) / sum_values) > p
                X(i,d) = 1;
            else
                X(i,d) = 0;
            end
        end
    end    
    % Fitness
    fit(i) = fun(feat,label,X(i,:) > thres,opts);
    % Feature vector (new)
    fv_n = X(i,:) > thres;
    % Feature vector (old)
    fv_o = Xpb(i,:) > thres;
    % Pbest update
    if fit(i) < fitP(i)
       increase_acc = fitP(i) - fit(i);
       Xpb(i,:) = X(i,:); 
       fitP(i)  = fit(i);
       % Changes in features (1:new features emerge, -1:old features disappear)
       change = fv_n - fv_o;
       case_emerge = find(change(1,:) == 1);
       case_disapper = find(change(1,:) == -1);
       % Calculate the weight of those features
       for j = 1 : length(case_emerge)
         weight(1, case_emerge(j)) = weight(1, case_emerge(j)) + increase_acc;
       end
       for j = 1 : length(case_disapper)
         weight(1, case_disapper(j)) = weight(1, case_disapper(j)) - increase_acc;
       end
       Xpb(i,:) = X(i,:); 
       fitP(i)  = fit(i);
    else
       decrease_acc = fit(i) - fitP(i);
       % Changes in features (1:old features, -1:new features)
       change = fv_o - fv_n;
       case_emerge = find(change(1,:) == -1);
       case_disapper = find(change(1,:) == 1);
       % Calculate the weight of those features
       for j = 1 : length(case_emerge)
           weight(1, case_emerge(j)) = weight(1, case_emerge(j)) - decrease_acc;
       end
       for j = 1 : length(case_disapper)
           weight(1, case_disapper(j)) = weight(1, case_disapper(j)) + decrease_acc;
       end
    end
    % SubBest update
    if fit(i) < fitSub(k)
       Xsub(k,:) = X(i,:);
       fitSub(k) = fit(i);
    end
    % Subpopulation update
    if mod(i,subSize) == 0     
        k = k + 1;
    end
    % Gbest update
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
  curve(t) = fitG; 
  fnum(t) = length(find((Xgb > thres) == 1));
  fprintf('\nIteration %d Best (MEL)= %f',t,curve(t))
  t = t + 1;
end
% save('weight.mat','weight');
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
PSO.sf = Sf; 
PSO.ff = sFeat;
PSO.nf = length(Sf);
fitGG = (fitG - 0.1*(PSO.nf / dim)) / 0.9; % real error rate
PSO.c  = curve;
PSO.fnum = fnum;
PSO.f  = feat;
PSO.l  = label;
PSO.fitG = fitGG;
end



