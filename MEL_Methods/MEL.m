% (14/12/2022)
% Developed by Xubin Wang, Email: wangxb19@mails.jlu.edu.cn

clear, clc, close;
numRun = 10;
for i=1:numRun
     Problem = {'Adenoma', 'ALL_AML', 'ALL3', 'ALL4', 'CNS', 'Colon', 'DLBCL', 'Gastric',...
                'Leukemia', 'Lymphoma', 'Prostate', 'Stroke'};%  
%    Problem = {'Stroke'};
    %% MAIN LOOP
    for j = 1:length(Problem)
        p_name = Problem{j};
        results = Training(p_name, i);
        results.p_name = p_name;                           
        saveResults(results);
    end 
end
toc

function results = Training(p_name, i)
    tic
    warning('off','all');
    traindata = load(['D:\KINDLAB\paper-7\data\',p_name]);
    traindata = getfield(traindata, p_name);
    data = traindata;
    feat = data(:,1:end-1); 
    label = data(:,end);

   %% Basic settings of PSO algorithm
    % Number of k in K-nearest neighbor
    opts.k = 3; 
    % Common parameter settings 
    opts.N  = 20;     % number of solutions
    T       = 100;    % number of iterations in each phase
    opts.thres = 0.6;
    opts.T  = T;
    PSO = jMultiTaskPSO(feat,label,opts);
    curve = PSO.c;
    fnum = PSO.fnum;
    filename = strcat('curve', p_name, num2str(i), '.mat');
    save(filename, 'curve');
    filename1 = strcat('fnum', p_name, num2str(i), '.mat');
    save(filename1, 'fnum');
    toc
    t = 1 - PSO.fitG;
    disp(t);
    results.optimized_Accuracy = 1 - PSO.fitG;
    results.selected_Features = PSO.nf;
    results.time = num2str(toc);
 end