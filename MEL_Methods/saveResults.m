function rs = saveResults(results)
  fid = [];
    if (exist([pwd filesep 'results.csv'], 'file') == 0)
        fid = fopen([pwd filesep 'results.csv'], 'w');
        fprintf(fid, '%s, %s, %s, %s\n', ...
            'Data Set','Avg Accuracy', 'Selected Features', 'Running Time');
    elseif (exist([pwd filesep 'results.csv'], 'file') == 2)
        fid = fopen([pwd filesep 'results.csv'], 'a');
    end
    fprintf(fid, '%s, ', results.p_name);
    fprintf(fid, '%f, %f, %s\n', ...
          results.optimized_Accuracy, results.selected_Features, results.time);
    fclose(fid);
end



