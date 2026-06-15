function plot_confusion_matrices(varargin)
% Plot aggregated row-normalized confusion matrices from saved CSV files.
%
% Example:
%   plot_confusion_matrices('outputFile','confusion_matrices.png')

p = inputParser;
addParameter(p, 'resultDirs', {"FS-C","results_final_eval/FS-C"; "IMG-D","results_final_eval/IMG-D"; "TL-A","results_final_eval/TL-A"}, @iscell);
addParameter(p, 'classLabels', ["ncm955_1","ncm955_2","ncm955_3","ncm955_4"], @(x) isstring(x) || iscellstr(x));
addParameter(p, 'outputFile', 'confusion_matrices_FS_IMG_TL.png', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});

resultDirs = p.Results.resultDirs;
classLabels = string(p.Results.classLabels);
outputFile = char(p.Results.outputFile);

figure('Position', [100, 100, 1400, 420]);
tiledlayout(1, size(resultDirs, 1), 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:size(resultDirs, 1)
    settingName = string(resultDirs{i, 1});
    folderPath = char(resultDirs{i, 2});
    C = aggregateConfusion(folderPath, numel(classLabels));
    Cnorm = C ./ max(sum(C, 2), eps);

    nexttile;
    imagesc(Cnorm);
    axis square;
    colormap(parula);
    clim([0 1]);
    colorbar;

    title(settingName, 'Interpreter', 'none');
    xlabel('Predicted class');
    ylabel('True class');
    xticks(1:numel(classLabels));
    yticks(1:numel(classLabels));
    xticklabels(classLabels);
    yticklabels(classLabels);
    xtickangle(45);

    for r = 1:size(Cnorm, 1)
        for c = 1:size(Cnorm, 2)
            text(c, r, sprintf('%.2f', Cnorm(r, c)), ...
                'HorizontalAlignment', 'center', ...
                'FontSize', 9, ...
                'Color', 'w');
        end
    end
end

sgtitle('Aggregated row-normalized confusion matrices across repeated held-out tests');
exportgraphics(gcf, outputFile, 'Resolution', 300);
fprintf('Saved figure to %s\n', outputFile);
end

function Csum = aggregateConfusion(folderPath, numClasses)
    files = dir(fullfile(folderPath, 'confusion_run_*_seed_*.csv'));
    if isempty(files)
        error('No confusion matrix CSV files found in %s.', folderPath);
    end

    Csum = zeros(numClasses);
    for i = 1:numel(files)
        Csum = Csum + readmatrix(fullfile(files(i).folder, files(i).name));
    end
end
