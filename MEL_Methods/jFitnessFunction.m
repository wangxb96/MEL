% Fitness Function KNN (9/12/2020)

function cost = jFitnessFunction(feat,label,X,opts)
% Default of [alpha; beta]
ws = [0.9; 0.1];  %[1; 0]; 

if isfield(opts,'ws'), ws = opts.ws; end

% Check if any feature exist
if sum(X == 1) == 0
  cost = 1;
else
  % Error rate
  error    = jwrapper_KNN(feat(:,X == 1),label,opts);
  % Number of selected features
  num_feat = sum(X == 1);
  % Total number of features
  max_feat = length(X); 
  % Set alpha & beta
  alpha    = ws(1); 
  beta     = ws(2);
  % Cost function 
  cost     = alpha * error + beta * (num_feat / max_feat); 
%    cost = error;
end
end


%---Call Functions-----------------------------------------------------
function error = jwrapper_KNN(sFeat,label,opts)
if isfield(opts,'k'), k = opts.k; end
Md = cvpartition(label, 'KFold', 5);
    for i = 1 : 5
        % Define training & validation sets
        testIdx = Md.test(i);
        xtrain   = sFeat(~testIdx,:); ytrain  = label(~testIdx);
        xvalid   = sFeat(testIdx,:);  yvalid  = label(testIdx);
        % Training model
        My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
        % Prediction
        pred     = predict(My_Model,xvalid);
        % Accuracy
        Acc(i)   = sum(pred == yvalid) / length(yvalid);
    end
% Error rate
error    = 1 - mean(Acc); 
end












