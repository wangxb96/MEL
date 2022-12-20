clear, clc, close;
numRun = 1;
for i=1:numRun
     Problem = {'curveColon', 'curveLeukemia', 'curveLymphoma', 'curveProstate'};%  
   % Problem = {'Prostate'};
    %% MAIN LOOP
    for j = 1:length(Problem)
        b = zeros(1, 500);
        for k = 1:5
            p_name = Problem{j};
            traindata = load(['E:\wxb''s notes\paper-6\FWPSO\Comparison_Methods\Other_ECs\ABC\',p_name, num2str(k)]);    
            a = getfield(traindata, 'curve'); 
            b = b + a;
        end 
        ABCcurve = b / 5;
        save(p_name, 'ABCcurve');
    end 
end
toc
