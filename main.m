function main(varargin)
% Final evaluation for NCM955 SEM calendering-state classification.
% Default: 10 repeated 70/10/20 train/validation/test splits.
%
% Examples:
%   main
%   main('dataDir','data','outputDir','results_final_eval')
%   main('runTransfer',false)

clearvars -except varargin; clc;
cfg = parseOptions(varargin{:});
ensureDir(cfg.outputDir);

diary(fullfile(cfg.outputDir, 'run_log.txt'));
diary on;
cleaner = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n==== Final NCM955 SEM CNN evaluation ====\n');
fprintf('Data: %s\nOutput: %s\n', cfg.dataDir, cfg.outputDir);
fprintf('Runs: %d | seeds: %s\n', cfg.numRuns, mat2str(cfg.seeds));
fprintf('Split: %.0f/%.0f/%.0f train/val/test\n', 100*cfg.trainRatio, 100*cfg.valRatio, 100*cfg.testRatio);

methodConfigs = finalMethodConfigs(cfg);
writetable(methodConfigs, fullfile(cfg.outputDir, 'final_method_configs.csv'));

data = imageDatastore(cfg.dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x) cropBottomTenth(imread(x)));

classNames = categories(data.Labels);
fprintf('\n==== Dataset summary ====\n');
disp(countEachLabel(data));
fprintf('Classes: %s\n', strjoin(string(classNames), ', '));

allRunMetrics = table();
allClassMetrics = table();

for i = 1:height(methodConfigs)
    setting = methodConfigs(i, :);
    [runTbl, classTbl] = runSetting(data, classNames, cfg, setting);
    allRunMetrics = [allRunMetrics; runTbl]; %#ok<AGROW>
    allClassMetrics = [allClassMetrics; classTbl]; %#ok<AGROW>
end

summaryTbl = summarizeRunMetrics(allRunMetrics);
classSummaryTbl = summarizeClassMetrics(allClassMetrics);

writetable(allRunMetrics, fullfile(cfg.outputDir, 'all_run_metrics.csv'));
writetable(allClassMetrics, fullfile(cfg.outputDir, 'all_class_metrics.csv'));
writetable(summaryTbl, fullfile(cfg.outputDir, 'summary_mean_sd.csv'));
writetable(classSummaryTbl, fullfile(cfg.outputDir, 'class_summary_mean_sd.csv'));

fprintf('\n==== Summary: mean +/- SD across runs ====\n');
disp(summaryTbl);
fprintf('\nSaved results to: %s\n', cfg.outputDir);
end

function cfg = parseOptions(varargin)
    p = inputParser;
    p.FunctionName = 'main';

    addParameter(p, 'dataDir', 'data', @(x) ischar(x) || isstring(x));
    addParameter(p, 'outputDir', 'results_final_eval', @(x) ischar(x) || isstring(x));
    addParameter(p, 'transferWeightsFile', 'pretrained_weights_from_e1.mat', @(x) ischar(x) || isstring(x));

    addParameter(p, 'numRuns', 10, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    addParameter(p, 'seedStart', 1, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'trainRatio', 0.70, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
    addParameter(p, 'valRatio', 0.10, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
    addParameter(p, 'testRatio', 0.20, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);

    addParameter(p, 'runFromScratch', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'runImagenet', true, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'runTransfer', true, @(x) islogical(x) && isscalar(x));

    addParameter(p, 'groupSplitByParent', false, @(x) islogical(x) && isscalar(x));
    addParameter(p, 'freezeTransferredLayers', false, @(x) islogical(x) && isscalar(x));

    parse(p, varargin{:});
    r = p.Results;

    if abs((r.trainRatio + r.valRatio + r.testRatio) - 1) > 1e-9
        error('trainRatio + valRatio + testRatio must equal 1.');
    end

    cfg = struct();
    cfg.dataDir = char(r.dataDir);
    cfg.outputDir = char(r.outputDir);
    cfg.transferWeightsFile = char(r.transferWeightsFile);
    cfg.numRuns = r.numRuns;
    cfg.seeds = r.seedStart:(r.seedStart + r.numRuns - 1);
    cfg.trainRatio = r.trainRatio;
    cfg.valRatio = r.valRatio;
    cfg.testRatio = r.testRatio;
    cfg.imageSize = [224 224 3];
    cfg.validationFrequency = 5;
    cfg.runFromScratch = r.runFromScratch;
    cfg.runImagenet = r.runImagenet;
    cfg.runTransfer = r.runTransfer;
    cfg.groupSplitByParent = r.groupSplitByParent;
    cfg.freezeTransferredLayers = r.freezeTransferredLayers;
    cfg.augmenter = imageDataAugmenter('RandRotation',[-10 10], 'RandXReflection',true, 'RandScale',[0.8 1.2]);
end

function methodConfigs = finalMethodConfigs(cfg)
    rows = {};
    if cfg.runFromScratch
        rows(end+1, :) = {'FS-C','from_scratch',3e-4,50,32,NaN,NaN}; %#ok<AGROW>
    end
    if cfg.runImagenet
        rows(end+1, :) = {'IMG-D','imagenet_pretrained',3e-4,50,32,NaN,NaN}; %#ok<AGROW>
    end
    if cfg.runTransfer
        rows(end+1, :) = {'TL-A','ncm_transfer',1e-4,20,32,1.0,10.0}; %#ok<AGROW>
    end
    if isempty(rows)
        error('No method selected. Enable at least one run flag.');
    end

    methodConfigs = cell2table(rows, 'VariableNames', ...
        {'setting','mode','initialLearnRate','maxEpochs','miniBatchSize','transferredLayerLRFactor','newClassifierLRFactor'});
    methodConfigs.setting = string(methodConfigs.setting);
    methodConfigs.mode = string(methodConfigs.mode);
end

function [runTbl, classTbl] = runSetting(data, classNames, cfg, setting)
    runTbl = table();
    classTbl = table();
    settingName = string(setting.setting);
    outDir = fullfile(cfg.outputDir, char(settingName));
    ensureDir(outDir);

    fprintf('\n==== %s | %s ====\n', settingName, string(setting.mode));
    fprintf('LR=%g | epochs=%d | batch=%d\n', setting.initialLearnRate, setting.maxEpochs, setting.miniBatchSize);

    for r = 1:cfg.numRuns
        seed = cfg.seeds(r);
        rng(seed, 'twister');
        fprintf('\n%s run %d/%d | seed=%d\n', settingName, r, cfg.numRuns, seed);

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
        [metrics, perClass, C] = evaluateModel(model, testAug, testSet.Labels, classNames);

        thisRun = table(settingName, string(setting.mode), r, seed, ...
            setting.initialLearnRate, setting.maxEpochs, setting.miniBatchSize, ...
            setting.transferredLayerLRFactor, setting.newClassifierLRFactor, ...
            numel(trainIdx), numel(valIdx), numel(testIdx), ...
            metrics.accuracy, metrics.macroPrecision, metrics.macroRecall, metrics.macroF1, ...
            'VariableNames', {'setting','mode','run','seed','initialLearnRate','maxEpochs','miniBatchSize', ...
            'transferredLayerLRFactor','newClassifierLRFactor','nTrain','nVal','nTest', ...
            'accuracy','macroPrecision','macroRecall','macroF1'});
        runTbl = [runTbl; thisRun]; %#ok<AGROW>

        perClass.setting = repmat(settingName, height(perClass), 1);
        perClass.mode = repmat(string(setting.mode), height(perClass), 1);
        perClass.run = repmat(r, height(perClass), 1);
        perClass.seed = repmat(seed, height(perClass), 1);
        perClass = movevars(perClass, {'setting','mode','run','seed'}, 'Before', 1);
        classTbl = [classTbl; perClass]; %#ok<AGROW>

        writematrix(C, fullfile(outDir, sprintf('confusion_run_%02d_seed_%d.csv', r, seed)));
        fprintf('Accuracy %.4f | Macro-F1 %.4f\n', metrics.accuracy, metrics.macroF1);
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
    preferred = {'netE1_old','modelE1','netE1','model','trainedNet','net'};
    for i = 1:numel(preferred)
        if isfield(S, preferred{i}) && isNetwork(S.(preferred{i}))
            oldNet = S.(preferred{i});
            return;
        end
    end
    names = fieldnames(S);
    for i = 1:numel(names)
        if isNetwork(S.(names{i}))
            oldNet = S.(names{i});
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

function summaryTbl = summarizeRunMetrics(runTbl)
    keys = unique(runTbl(:, {'setting','mode'}), 'rows', 'stable');
    rows = table();
    for i = 1:height(keys)
        mask = runTbl.setting == keys.setting(i) & runTbl.mode == keys.mode(i);
        sub = runTbl(mask, :);
        row = table(keys.setting(i), keys.mode(i), height(sub), ...
            sub.initialLearnRate(1), sub.maxEpochs(1), sub.miniBatchSize(1), ...
            sub.transferredLayerLRFactor(1), sub.newClassifierLRFactor(1), ...
            mean(sub.accuracy), std(sub.accuracy), ...
            mean(sub.macroPrecision), std(sub.macroPrecision), ...
            mean(sub.macroRecall), std(sub.macroRecall), ...
            mean(sub.macroF1), std(sub.macroF1), ...
            'VariableNames', {'setting','mode','nRuns','initialLearnRate','maxEpochs','miniBatchSize', ...
            'transferredLayerLRFactor','newClassifierLRFactor','accuracy_mean','accuracy_sd', ...
            'macroPrecision_mean','macroPrecision_sd','macroRecall_mean','macroRecall_sd','macroF1_mean','macroF1_sd'});
        rows = [rows; row]; %#ok<AGROW>
    end
    summaryTbl = rows;
end

function summaryTbl = summarizeClassMetrics(classTbl)
    keys = unique(classTbl(:, {'setting','mode','class'}), 'rows', 'stable');
    rows = table();
    for i = 1:height(keys)
        mask = classTbl.setting == keys.setting(i) & classTbl.mode == keys.mode(i) & classTbl.class == keys.class(i);
        sub = classTbl(mask, :);
        row = table(keys.setting(i), keys.mode(i), keys.class(i), height(sub), ...
            mean(sub.precision), std(sub.precision), mean(sub.recall), std(sub.recall), mean(sub.f1), std(sub.f1), ...
            'VariableNames', {'setting','mode','class','nRuns','precision_mean','precision_sd','recall_mean','recall_sd','f1_mean','f1_sd'});
        rows = [rows; row]; %#ok<AGROW>
    end
    summaryTbl = rows;
end

function img = cropBottomTenth(img)
    img = img(1:round(size(img, 1) * 0.9), :, :);
end

function ensureDir(pathName)
    if ~exist(pathName, 'dir')
        mkdir(pathName);
    end
end
