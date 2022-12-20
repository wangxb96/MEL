clear, clc, close;
numRun = 10;
for i=1:numRun
     Problem = {'ALL_AML', 'ALL3', 'ALL4', 'CNS', 'Colon', 'DLBCL', 'Gastric',...
                'Leukemia', 'Lymphoma', 'Myeloma', 'Prostate'};%  , 'Stroke'
%    Problem = {'Stroke'};
    %% MAIN LOOP
    for j = 1:length(Problem)
        for eachClass = 1:1
            p_name = Problem{j};
            results = Training(p_name, i);
            results.p_name = p_name;                           
            saveResults(results);
        end 
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

   %% Basic settings EC algorithms
    % Number of k in K-nearest neighbor
    opts.k = 3; 
    % Common parameter settings 
    opts.N  = 20;     % number of solutions
    T       = 100;    % number of iterations in each phase
    opts.thres = 0.6;
    opts.T  = T;    % maximum number of iterations

   %% Swarm-based  
%    %% Feature selection using ACO algorithm
%      Temp = jAntColonyOptimization(feat,label,opts);
%    
%    %% Feature selection using ABC algorithm
%    Temp = jArtificialBeeColony(feat,label,opts);
% 
%     %% Feature selection using PSO algorithm
%     Temp = jParticleSwarmOptimization(feat,label,opts);  
% 
%     %% Feature selection using MBO algorithm
%      Temp = jMonarchButterflyOptimization(feat,label,opts);

    %% Nature-inspired 
%     %% Feature selection using BAT algorithm
%     Temp = jBatAlgorithm(feat,label,opts);

%    %% Feature selection using CS algorithm
%     Temp = jCuckooSearchAlgorithm(feat,label,opts);

%    %% Feature selection using FA algorithm
%     Temp = jFireflyAlgorithm(feat,label,opts);

%    %% Feature selection using FPA algorithm
%     Temp = jFlowerPollinationAlgorithm(feat,label,opts);

    %% Evolutionary Algorithms
%    %% Feature selection using DE algorithm
%     Temp = jDifferentialEvolution(feat,label,opts);     

%    %% Feature selection using GA algorithm
%     Temp = jGeneticAlgorithm(feat,label,opts);

%    %% Feature selection using GA (Tour) algorithm
    Temp = jGeneticAlgorithmTour(feat,label,opts);

    %% Bio-stimulated
%    %% Feature selection using FFO algorithm
%     Temp = jFruitFlyOptimizationAlgorithm(feat,label,opts);
% 
%    %% Feature selection using GWO algorithm
%     Temp = jGreyWolfOptimizer(feat,label,opts);
%    
%    %% Feature selection using HHO algorithm
%     Temp = jHarrisHawksOptimization(feat,label,opts);
%    
%    %% Feature selection using WHO algorithm
%     Temp = jWhaleOptimizationAlgorithm(feat,label,opts);

    %% Physics-based
%    %% Feature selection using SA algorithm
%     Temp = jSimulatedAnnealing(feat,label,opts);

%    %% Feature selection using HS algorithm
%     Temp = jHarmonySearch(feat,label,opts);

%    %% Feature selection using GSA algorithm
%     Temp = jGravitationalSearchAlgorithm(feat,label,opts);

%    %% Feature selection using MVO algorithm
%     Temp = jMultiVerseOptimizer(feat,label,opts);

    curve = Temp.c;
    fnum = Temp.fnum;
    
    filename = strcat('curve', p_name, num2str(i), '.mat');
    save(filename, 'curve');
    filename1 = strcat('fnum', p_name, num2str(i), '.mat');
    save(filename1, 'fnum');
    toc
    
    fitG = curve(:,end);
    results.optimized_Accuracy = 1- fitG;
    results.selected_Features = Temp.nf;
    results.time = num2str(toc);
 end




