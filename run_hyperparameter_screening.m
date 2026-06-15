function run_hyperparameter_screening(varargin)
% Hyperparameter screening for NCM955 SEM calendering-state classification.
% Settings are selected by validation macro-F1, not by held-out test metrics.
%
% Examples:
%   run_hyperparameter_screening
%   run_hyperparameter_screening('runOnlyMode','imagenet_pretrained')
%   run_hyperparameter_screening('runOnlySettings',["IMG-C","IMG-D"])

clearvars -except varargin; clc;
cfg = parseOptions(varargin{:});
ensureDir(cfg.outputDir);

diary(fullfile(cfg.outputDir, 'run_log.txt'));
diary on;
cleaner = onCleanup(@() diary('off')); %#ok<NASGU>

grid = makeGrid(cfg);
writetable(grid, fullfile(cfg.outputDir, 'screening_grid.csv'));

data = imageDatastore(cfg.dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x) cropBottomTenth(imread(x)));

classNames = categories(data.Labels);
fprintf('\n==== Dataset summary ====\n');
fprintf('Data: %s\nOutput: %s\n', cfg.dataDir, cfg.outputDir);
disp(countEachLabel(data));
fprintf('Classes: %s\n', strjoin(string(classNames), ', '));
fprintf('Screening seeds: %s\n', mat2str(cfg.seeds));

allMetrics = table();
allClassMetrics = table();

for i = 1:height(grid)
    setting = grid(i, :);
    [runTbl, classTbl] = runSetting(data, classNames, cfg, setting);
    allMetrics = [allMetrics; runTbl]; %#ok<AGROW>
    allClassMetrics = [allClassMetrics; classTbl]; %#ok<AGROW>
end

summaryTbl = summarizeScreeningMetrics(allMetrics);
bestTbl = selectBestByValidation(summaryTbl);

writetable(allMetrics, fullfile(cfg.outputDir, 'all_screening_metrics.csv'));
writetable(allClassMetrics, fullfile(cfg.outputDir, 'all_screening_class_metrics.csv'));
writetable(summaryTbl, fullfile(cfg.outputDir, 'screening_summary_mean_sd.csv'));
writetable(bestTbl, fullfile(cfg.outputDir, 'best_by_validation.csv'));

fprintf('\n==== Screening summary ====\n');
disp(summaryTbl);
fprintf('\n==== Best setting by validation macro-F1 ====\n');
disp(bestTbl);
end

function cfg = parseOptions(varargin)
    p = inputParser;
    p.FunctionName = 'run_hyperparameter_screening';

    addParameter(p, 'dataDir', 'data', @(x) ischar(x) || isstring(x));
    addParameter(p, 'outputDir', 'results_hyperparam_screening', @(x) ischar(x) || isstring(x));
    addParameter(p, 'transferWeightsFile', 'pretrained_weights_from_e1.mat', @(x) ischar(x) || isstring(x));
    addParameter(p, 'runOnlyMode', "", @(x) ischar(x) || isstring(x));
    addParameter(p, 'runOnlySettings', strings(0,1), @(x) isstring(x) || iscellstr(x));
    addParameter(p, 'numRuns', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'seedStart', 1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'groupSplitByParent', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'freezeTransferredLayers', false, @(x) islogical(x) && isscalar(x));

    parse(p, varargin{:});
    r = p.Results;

    cfg = struct();
    cfg.dataDir = char(r.dataDir);
    cfg.outputDir = char(r.outputDir);
    cfg.transferWeightsFile = char(r.transferWeightsFile);
    cfg.runOnlyMode = string(r.runOnlyMode);
    cfg.runOnlySettings = string(r.runOnlySettings(:));
    cfg.numRuns = r.numRuns;
    cfg.seeds = r.seedStart:(r.seedStart + r.numRuns - 1);
    cfg.trainRatio = 0.70;
    cfg.valRatio = 0.10;
    cfg.testRatio = 0.20;
    cfg.imageSize = [224 224 3];
    cfg.validationFrequency = 5;
    cfg.groupSplitByParent = r.groupSplitByParent;
    cfg.freezeTransferredLayers = r.freezeTransferredLayers;
    cfg.augmenter = imageDataAugmenter('RandRotation',[-10 10], 'RandXReflection',true, 'RandScale',[0.8 1.2]);

    if strlength(cfg.runOnlyMode) > 0 && strcmp(r.outputDir, 'results_hyperparam_screening')
        cfg.outputDir = char("results_hyperparam_screening_" + cfg.runOnlyMode + "_only");
    end
end

function grid = makeGrid(cfg)
    rows = {
        'FS-A','from_scratch',1e-4,20,32,NaN,NaN;
        'FS-B','from_scratch',1e-4,50,32,NaN,NaN;
        'FS-C','from_scratch',3e-4,50,32,NaN,NaN;
        'FS-D','from_scratch',3e-4,50,16,NaN,NaN;
        'IMG-A','imagenet_pretrained',1e-4,20,32,NaN,NaN;
        'IMG-B','imagenet_pretrained',1e-4,50,32,NaN,NaN;
        'IMG-C','imagenet_pretrained',3e-5,50,32,NaN,NaN;
        'IMG-D','imagenet_pretrained',3e-4,50,32,NaN,NaN;
        'TL-A','ncm_transfer',1e-4,20,32,1.0,10.0;
        'TL-B','ncm_transfer',1e-5,50,32,1.0,10.0;
        'TL-C','ncm_transfer',1e-4,50,32,0.1,10.0;
        'TL-D','ncm_transfer',1e-5,50,32,1.0,20.0};

    grid = cell2table(rows, 'VariableNames', ...
        {'setting','mode','initialLearnRate','maxEpochs','miniBatchSize','transferredLayerLRFactor','newClassifierLRFactor'});
    grid.setting = string(grid.setting);
    grid.mode = string(grid.mode);

    if strlength(cfg.runOnlyMode) > 0
        grid = grid(grid.mode == cfg.runOnlyMode, :);
    end
    if ~isempty(cfg.runOnlySettings)
        grid = grid(ismember(grid.setting, cfg.runOnlySettings), :);
    end
    if isempty(grid)
        error('No settings remain after filtering.');
    end
end

function [runTbl, classTbl] = runSetting(data, classNames, cfg, setting)
    runTbl = table();
    classTbl = table();
    outDir = fullfile(cfg.outputDir, char(setting.setting));
    ensureDir(outDir);

    fprintf('\n==== %s | %s ====\n', setting.setting, setting.mode);
    fprintf('LR=%g | epochs=%d | batch=%d\n', setting.initialLearnRate, setting.maxEpochs, setting.miniBatchSize);

    for r = 1:cfg.numRuns
        seed = cfg.seeds(r);
        rng(seed, 'twister');
        fprintf('\n%s run %d/%d | seed=%d\n', setting.setting, r, cfg.numRuns, seed);

        [trainIdx, valIdx, testIdx] = makeSplit(data, cfg, seed);
        trainSet = subset(data, trainIdx);
        valSet = subset(data, valIdx);
        testSet = subset(data, testIdx);

        trainAug = augmentedImageDatastore(cfg.imageSize, trainSet, 'DataAugmentation', cfg.augmenter, 'ColorPreprocessing', 'gray2rgb');
        valAug = augmentedImageDatastore(cfg.imageSize, valSet, 'ColorPreprocessing', 'gray2rgb');
        testAug = augmentedImageDatastore(cfg.imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

        net = buildNetwork(setting, numel(classNames), cfg);
        opts = trainingOptions('adam', ...
            'MaxEpochs', setting.maxEpochs, ...
            'MiniBatchSize', setting.miniBatchSize, ...
            'InitialLearnRate', setting.initialLearnRate, ...
            'Metrics', 'accuracy', ...
            'ValidationData', valAug, ...
            'ValidationFrequency', cfg.validationFrequency, ...
            'Verbose', false, ...
            'Plots', 'none');

        model = trainnet(trainAug, net, 'crossentropy', opts);
        [valMetrics, valClass] = evaluateModel(model, valAug, valSet.Labels, classNames);
        [testMetrics, testClass, C] = evaluateModel(model, testAug, testSet.Labels, classNames);

        thisRun = table(setting.setting, setting.mode, r, seed, ...
            setting.initialLearnRate, setting.maxEpochs, setting.miniBatchSize, ...
            setting.transferredLayerLRFactor, setting.newClassifierLRFactor, ...
            numel(trainIdx), numel(valIdx), numel(testIdx), ...
            valMetrics.accuracy, valMetrics.macroPrecision, valMetrics.macroRecall, valMetrics.macroF1, ...
            testMetrics.accuracy, testMetrics.macroPrecision, testMetrics.macroRecall, testMetrics.macroF1, ...
            'VariableNames', {'setting','mode','run','seed','initialLearnRate','maxEpochs','miniBatchSize', ...
            'transferredLayerLRFactor','newClassifierLRFactor','nTrain','nVal','nTest', ...
            'val_accuracy','val_macroPrecision','val_macroRecall','val_macroF1', ...
            'test_accuracy','test_macroPrecision','test_macroRecall','test_macroF1'});
        runTbl = [runTbl; thisRun]; %#ok<AGROW>

        valClass.split = repmat("validation", height(valClass), 1);
        testClass.split = repmat("test", height(testClass), 1);
        perClass = [valClass; testClass];
        perClass.setting = repmat(setting.setting, height(perClass), 1);
        perClass.mode = repmat(setting.mode, height(perClass), 1);
        perClass.run = repmat(r, height(perClass), 1);
        perClass.seed = repmat(seed, height(perClass), 1);
        perClass = movevars(perClass, {'setting','mode','run','seed','split'}, 'Before', 1);
        classTbl = [classTbl; perClass]; %#ok<AGROW>

        writematrix(C, fullfile(outDir, sprintf('confusion_run_%02d_seed_%d.csv', r, seed)));
        fprintf('Val macro-F1 %.4f | Test macro-F1 %.4f\n', valMetrics.macroF1, testMetrics.macroF1);
    end
end

function net = buildNetwork(setting, numClasses, cfg)
    switch string(setting.mode)
        case 'from_scratch'
            net = imagePretrainedNetwork('efficientnetb0', 'NumClasses', numClasses, 'Weights', 'none');
        case 'imagenet_pretrained'
            net = imagePretrainedNetwork('efficientnetb0', 'NumClasses', numClasses);
        case 'ncm_transfer'
            oldNet = loadTransferNetwork(cfg.transferWeightsFile);
            net = replaceClassifier(oldNet, numClasses, cfg, setting.transferredLayerLRFactor, setting.newClassifierLRFactor);
        otherwise
            error('Unknown mode: %s', string(setting.mode));
    end
end

function oldNet = loadTransferNetwork(fileName)
    if ~exist(fileName, 'file')
        error('Transfer weights file not found: %s', fileName);
    end
    S = load(fileName);
    names = [{'netE1_old','modelE1','netE1','model','trainedNet','net'}, fieldnames(S)'];
    for i = 1:numel(names)
        name = names{i};
        if isfield(S, name) && isNetwork(S.(name))
            oldNet = S.(name);
            return;
        end
    end
    error('No supported network object found in %s.', fileName);
end

function tf = isNetwork(x)
    tf = isa(x, 'dlnetwork') || isa(x, 'DAGNetwork') || isa(x, 'SeriesNetwork');
end

function net = replaceClassifier(oldNet, numClasses, cfg, oldFactor, newFactor)
    lgraph = removeOutputLayers(layerGraph(oldNet));
    layers = lgraph.Layers;

    for i = 1:numel(layers)
        lrFactor = oldFactor;
        if cfg.freezeTransferredLayers
            lrFactor = 0;
        end
        layer = setLRFactor(layers(i), lrFactor);
        try
            lgraph = replaceLayer(lgraph, layer.Name, layer);
        catch
        end
    end

    layers = lgraph.Layers;
    fcIdx = find(arrayfun(@(x) isa(x, 'nnet.cnn.layer.FullyConnectedLayer'), layers), 1, 'last');
    if isempty(fcIdx)
        error('No fullyConnectedLayer found in transfer network.');
    end

    fcName = layers(fcIdx).Name;
    newFC = fullyConnectedLayer(numClasses, 'Name', fcName, ...
        'WeightLearnRateFactor', newFactor, 'BiasLearnRateFactor', newFactor);
    net = dlnetwork(replaceLayer(lgraph, fcName, newFC));
end

function lgraph = removeOutputLayers(lgraph)
    layers = lgraph.Layers;
    names = string({layers.Name});
    isOutput = arrayfun(@(x) isa(x, 'nnet.cnn.layer.ClassificationOutputLayer') || isa(x, 'nnet.cnn.layer.SoftmaxLayer'), layers);
    if any(isOutput)
        lgraph = removeLayers(lgraph, cellstr(names(isOutput)));
    end
end

function layer = setLRFactor(layer, value)
    props = {'WeightLearnRateFactor','BiasLearnRateFactor','ScaleLearnRateFactor','OffsetLearnRateFactor'};
    for i = 1:numel(props)
        if isprop(layer, props{i})
            layer.(props{i}) = value;
        end
    end
end

function [trainIdx, valIdx, testIdx] = makeSplit(data, cfg, seed)
    if cfg.groupSplitByParent
        [trainIdx, valIdx, testIdx] = stratifiedGroupSplit(data, cfg, seed);
    else
        [trainIdx, valIdx, testIdx] = stratifiedImageSplit(data, cfg, seed);
    end
end

function [trainIdx, valIdx, testIdx] = stratifiedImageSplit(data, cfg, seed)
    rng(seed, 'twister');
    labels = data.Labels;
    classes = categories(labels);
    trainIdx = []; valIdx = []; testIdx = [];
    for i = 1:numel(classes)
        idx = find(labels == classes{i});
        idx = idx(randperm(numel(idx)));
        nTrain = floor(cfg.trainRatio * numel(idx));
        nVal = floor(cfg.valRatio * numel(idx));
        trainIdx = [trainIdx; idx(1:nTrain)]; %#ok<AGROW>
        valIdx = [valIdx; idx(nTrain+1:nTrain+nVal)]; %#ok<AGROW>
        testIdx = [testIdx; idx(nTrain+nVal+1:end)]; %#ok<AGROW>
    end
    trainIdx = trainIdx(randperm(numel(trainIdx)));
    valIdx = valIdx(randperm(numel(valIdx)));
    testIdx = testIdx(randperm(numel(testIdx)));
end

function [trainIdx, valIdx, testIdx] = stratifiedGroupSplit(data, cfg, seed)
    rng(seed, 'twister');
    labels = data.Labels;
    files = data.Files;
    classes = categories(labels);
    parentIDs = strings(numel(files), 1);
    for i = 1:numel(files)
        parentIDs(i) = parentIdFromFilename(files{i});
    end

    trainIdx = []; valIdx = []; testIdx = [];
    for i = 1:numel(classes)
        classIdx = find(labels == classes{i});
        groups = unique(parentIDs(classIdx));
        groups = groups(randperm(numel(groups)));
        nTrain = max(1, floor(cfg.trainRatio * numel(groups)));
        nVal = max(1, floor(cfg.valRatio * numel(groups)));
        if nTrain + nVal >= numel(groups)
            error('Class %s has too few parent groups for group splitting.', classes{i});
        end
        trainGroups = groups(1:nTrain);
        valGroups = groups(nTrain+1:nTrain+nVal);
        testGroups = groups(nTrain+nVal+1:end);
        trainIdx = [trainIdx; classIdx(ismember(parentIDs(classIdx), trainGroups))]; %#ok<AGROW>
        valIdx = [valIdx; classIdx(ismember(parentIDs(classIdx), valGroups))]; %#ok<AGROW>
        testIdx = [testIdx; classIdx(ismember(parentIDs(classIdx), testGroups))]; %#ok<AGROW>
    end
end

function parentID = parentIdFromFilename(filePath)
    [~, name, ~] = fileparts(filePath);
    parts = split(string(name), '_');
    parentID = parts(1);
end

function [metrics, perClassTbl, C] = evaluateModel(model, augData, trueLabels, classNames)
    scores = minibatchpredict(model, augData);
    predLabels = scores2label(scores, classNames);
    C = confusionmat(trueLabels, predLabels, 'Order', categorical(classNames));

    n = numel(classNames);
    precision = zeros(n, 1);
    recall = zeros(n, 1);
    f1 = zeros(n, 1);
    for i = 1:n
        tp = C(i, i);
        fp = sum(C(:, i)) - tp;
        fn = sum(C(i, :)) - tp;
        precision(i) = safeDivide(tp, tp + fp);
        recall(i) = safeDivide(tp, tp + fn);
        f1(i) = safeDivide(2 * precision(i) * recall(i), precision(i) + recall(i));
    end

    metrics.accuracy = safeDivide(sum(diag(C)), sum(C(:)));
    metrics.macroPrecision = mean(precision);
    metrics.macroRecall = mean(recall);
    metrics.macroF1 = mean(f1);
    perClassTbl = table(string(classNames(:)), precision, recall, f1, 'VariableNames', {'class','precision','recall','f1'});
end

function y = safeDivide(a, b)
    if b == 0
        y = 0;
    else
        y = a / b;
    end
end

function summaryTbl = summarizeScreeningMetrics(metricsTbl)
    keys = unique(metricsTbl(:, {'setting','mode'}), 'rows', 'stable');
    rows = table();
    for i = 1:height(keys)
        mask = metricsTbl.setting == keys.setting(i) & metricsTbl.mode == keys.mode(i);
        sub = metricsTbl(mask, :);
        row = table(keys.setting(i), keys.mode(i), height(sub), ...
            sub.initialLearnRate(1), sub.maxEpochs(1), sub.miniBatchSize(1), ...
            sub.transferredLayerLRFactor(1), sub.newClassifierLRFactor(1), ...
            mean(sub.val_macroF1), std(sub.val_macroF1), ...
            mean(sub.test_macroF1), std(sub.test_macroF1), ...
            mean(sub.val_accuracy), std(sub.val_accuracy), ...
            mean(sub.test_accuracy), std(sub.test_accuracy), ...
            'VariableNames', {'setting','mode','nRuns','initialLearnRate','maxEpochs','miniBatchSize', ...
            'transferredLayerLRFactor','newClassifierLRFactor','val_macroF1_mean','val_macroF1_sd', ...
            'test_macroF1_mean','test_macroF1_sd','val_accuracy_mean','val_accuracy_sd','test_accuracy_mean','test_accuracy_sd'});
        rows = [rows; row]; %#ok<AGROW>
    end
    summaryTbl = rows;
end

function bestTbl = selectBestByValidation(summaryTbl)
    modes = unique(summaryTbl.mode, 'stable');
    bestTbl = table();
    for i = 1:numel(modes)
        sub = summaryTbl(summaryTbl.mode == modes(i), :);
        [~, idx] = max(sub.val_macroF1_mean);
        bestTbl = [bestTbl; sub(idx, :)]; %#ok<AGROW>
    end
end

function img = cropBottomTenth(img)
    img = img(1:round(size(img, 1) * 0.9), :, :);
end

function ensureDir(pathName)
    if ~exist(pathName, 'dir')
        mkdir(pathName);
    end
end
