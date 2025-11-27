%% LFT-Net (Proposed) vs MPT-SFANet (TAES 2024 Baseline)
% 目标：全维度对比实验 (Acc, Precision, Recall, F1)
clear; close all; clc;
rng(42);

%% 1. 参数设置
fprintf('======================================================\n');
fprintf('  LFT-Net vs MPT-SFANet 全维度对比实验 (TAES标准)\n');
fprintf('======================================================\n\n');

numClasses = 9;
numTrainSamplesPerClass = 600;
numTestSamplesPerClass = 200;
numFeatures = 10;
T = 10;

%% 2. 数据准备
fprintf('【步骤1】生成平衡雷达数据...\n');
[X_train, Y_train] = generateBalancedRadarData(numClasses, numTrainSamplesPerClass, numFeatures, T);
[X_test, Y_test] = generateBalancedRadarData(numClasses, numTestSamplesPerClass, numFeatures, T);

% 格式转换
X_train_seq = cell(size(X_train, 3), 1);
X_test_seq = cell(size(X_test, 3), 1);
for i = 1:size(X_train, 3), X_train_seq{i} = X_train(:, :, i)'; end
for i = 1:size(X_test, 3), X_test_seq{i} = X_test(:, :, i)'; end

% 静态特征 (用于SVM/MLP)
X_train_static = squeeze(mean(X_train, 2))';
X_test_static = squeeze(mean(X_test, 2))';

%% 3. 模型定义与训练 (带进度条)
fig = figure('Name', 'Training Monitor');

% 通用训练选项
getOpts = @(lr) trainingOptions('adam', ...
    'MaxEpochs', 150, 'MiniBatchSize', 64, ...
    'InitialLearnRate', lr, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 40, ...
    'ValidationData', {X_test_seq, Y_test}, 'ValidationFrequency', 30, ...
    'Plots', 'training-progress', 'Verbose', false);

% 1. SVM
fprintf('=== 1/6 训练 SVM (Static Baseline) ===\n');
updateProgress(fig, 1, 6, 'SVM');
svm_template = templateSVM('KernelFunction','linear', 'BoxConstraint', 0.1, 'KernelScale', 'auto');
svm_mdl = fitcecoc(X_train_static, Y_train, 'Learners', svm_template, 'Coding', 'onevsall');
Y_pred_svm = predict(svm_mdl, X_test_static);

% 2. MLP
fprintf('=== 2/6 训练 MLP (NN Baseline) ===\n');
updateProgress(fig, 2, 6, 'MLP');
layers_mlp = [
    featureInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'zscore')
    fullyConnectedLayer(128, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.4, 'Name', 'drop1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.4, 'Name', 'drop2')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    ];
opts_mlp = trainingOptions('adam', ...
    'MaxEpochs', 120, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.003, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 40, ...
    'L2Regularization', 0.0001, ...
    'GradientThreshold', 2, ...
    'ValidationData', {X_test_static, Y_test}, ...
    'ValidationFrequency', 30, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);
[net_mlp, info_mlp] = trainnet(X_train_static, Y_train, layers_mlp, 'crossentropy', opts_mlp);
Y_scores_mlp = predict_with_format(net_mlp, X_test_static);
Y_pred_mlp = scores2label(Y_scores_mlp, categories(Y_train));

% 3. LSTM
fprintf('=== 3/6 训练 LSTM (RNN Baseline) ===\n');
updateProgress(fig, 3, 6, 'LSTM');
layers_lstm = [
    sequenceInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'zscore')
    lstmLayer(90, 'OutputMode', 'sequence', 'Name', 'lstm1')
    dropoutLayer(0.4, 'Name', 'drop1')
    lstmLayer(60, 'OutputMode', 'last', 'Name', 'lstm2')
    dropoutLayer(0.3, 'Name', 'drop2')
    fullyConnectedLayer(16, 'Name', 'fc_mid')
    batchNormalizationLayer('Name', 'bn_mid')
    reluLayer('Name', 'relu_mid')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    ];
opts_lstm = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.002, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 30, ...
    'L2Regularization', 0.0005, ...
    'GradientThreshold', 1, ...
    'ValidationData', {X_test_seq, Y_test}, ...
    'ValidationFrequency', 30, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);
[net_lstm, info_lstm] = trainnet(X_train_seq, Y_train, layers_lstm, 'crossentropy', opts_lstm);
Y_scores_lstm = predict_with_format(net_lstm, X_test_seq);
Y_pred_lstm = scores2label(Y_scores_lstm, categories(Y_train));

% 4. GRU
fprintf('=== 4/6 训练 GRU (RNN Baseline) ===\n');
updateProgress(fig, 4, 6, 'GRU');
layers_gru = [
    sequenceInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'zscore')
    gruLayer(84, 'OutputMode', 'sequence', 'Name', 'gru1')
    dropoutLayer(0.3, 'Name', 'drop1')
    gruLayer(48, 'OutputMode', 'last', 'Name', 'gru2')
    dropoutLayer(0.3, 'Name', 'drop2')
    fullyConnectedLayer(12, 'Name', 'fc_mid')
    batchNormalizationLayer('Name', 'bn_mid')
    reluLayer('Name', 'relu_mid')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    ];
opts_gru = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.002, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 30, ...
    'L2Regularization', 0.0005, ...
    'GradientThreshold', 1, ...
    'ValidationData', {X_test_seq, Y_test}, ...
    'ValidationFrequency', 30, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);
[net_gru, info_gru] = trainnet(X_train_seq, Y_train, layers_gru, 'crossentropy', opts_gru);
Y_scores_gru = predict_with_format(net_gru, X_test_seq);
Y_pred_gru = scores2label(Y_scores_gru, categories(Y_train));

% 5. MPT-SFANet (Comparison)
fprintf('=== 5/6 训练 MPT-SFANet (TAES Ref) ===\n');
updateProgress(fig, 5, 6, 'MPT-SFANet');
layers_mpt = defineMPT_SFANet_1D(numFeatures, numClasses);
net_mpt = trainNetwork(X_train_seq, Y_train, layers_mpt, getOpts(0.001));
Y_pred_mpt = classify(net_mpt, X_test_seq);

% 6. LFT-Net (Proposed)
fprintf('=== 6/6 训练 LFT-Net (Ours) ===\n');
updateProgress(fig, 6, 6, 'LFT-Net');
layers_lft = defineLFTNet(numFeatures, numClasses);
net_lft = trainNetwork(X_train_seq, Y_train, layers_lft, getOpts(0.001));
Y_pred_lft = classify(net_lft, X_test_seq);

%% 4. 结果可视化与全指标对比

fprintf('\n=============================================================================\n');
fprintf('                        实验结果对比 (Experimental Results)\n');
fprintf('=============================================================================\n');
fprintf('%-15s |  Accuracy  |  Precision |   Recall   |  F1-Score  | Params(M)\n', 'Method');
fprintf('-----------------------------------------------------------------------------\n');

% 定义模型列表
models_name = {'SVM', 'MLP', 'LSTM', 'GRU', 'MPT-SFANet', 'LFT-Net'};
preds_list = {Y_pred_svm, Y_pred_mlp, Y_pred_lstm, Y_pred_gru, Y_pred_mpt, Y_pred_lft};
% 估算参数量 (M) - 这里仅为示例值，具体可由 analyzeNetwork 计算
params_list = [0.01, 0.02, 0.05, 0.04, 0.11, 0.12]; 

results_matrix = zeros(6, 4); % 存储数值用于后续画图

for i = 1:length(models_name)
    % 计算四个指标
    [acc, prec, rec, f1] = calculateAllMetrics(preds_list{i}, Y_test);
    results_matrix(i, :) = [acc, prec, rec, f1];
    
    % 打印表格行
    fprintf('%-15s |   %5.2f%%   |   %5.2f%%   |   %5.2f%%   |   %5.2f%%   |   %.2f\n', ...
        models_name{i}, acc*100, prec*100, rec*100, f1*100, params_list(i));
end
fprintf('=============================================================================\n');

%% 4.5 测量推理时间 (Inference Time Measurement)
fprintf('\n======================================\n');
fprintf('  【步骤 4.5】测量单样本推理时间\n');
fprintf('======================================\n');

num_trials = 100;  % 重复100次取平均
warmup_runs = 10;  % 预热次数

% 准备单个测试样本
test_sample_static = X_test_static(1, :);  % 用于 SVM, MLP
test_sample_seq = X_test_seq(1);           % 用于 LSTM, GRU, MPT, LFT (Cell格式)

% 定义一个通用的测量函数句柄 (避免重复写循环)
measure_time = @(model, data, type) measure_inference_internal(model, data, type, num_trials, warmup_runs);

% 1. SVM 推理时间
fprintf('  1. 测量 SVM...\n');
t_svm = measure_time(svm_mdl, test_sample_static, 'predict');

% 2. MLP 推理时间
fprintf('  2. 测量 MLP...\n');
t_mlp = measure_time(net_mlp, test_sample_static, 'predict');

% 3. LSTM 推理时间
fprintf('  3. 测量 LSTM...\n');
t_lstm = measure_time(net_lstm, test_sample_seq, 'predict');

% 4. GRU 推理时间
fprintf('  4. 测量 GRU...\n');
t_gru = measure_time(net_gru, test_sample_seq, 'predict');

% 5. MPT-SFANet 推理时间
fprintf('  5. 测量 MPT-SFANet...\n');
t_mpt = measure_time(net_mpt, test_sample_seq, 'classify');

% 6. LFT-Net 推理时间
fprintf('  6. 测量 LFT-Net...\n');
t_lft = measure_time(net_lft, test_sample_seq, 'classify');

% 汇总时间数据
times_list = [t_svm, t_mlp, t_lstm, t_gru, t_mpt, t_lft];
fprintf('  测量完成。\n');

%% 4.8 数据量敏感性实验（少样本/数据效率）
% fprintf('\n======================================\n');
% fprintf('  【步驟 4.8】数据量敏感性 (少样本/数据效率)\n');
% fprintf('======================================\n');

% sample_ratios = [0.25, 0.50, 0.75, 1.00];  % 训练样本占比
% numSeeds = 3;                               % 重复次数，统计均值方差
% sens_models = {'LSTM', 'MPT-SFANet', 'LFT-Net'}; % 关注的代表性模型

% acc_stack = zeros(numel(sens_models), numel(sample_ratios), numSeeds);
% f1_stack = zeros(numel(sens_models), numel(sample_ratios), numSeeds);

% for r = 1:numel(sample_ratios)
%     ratio = sample_ratios(r);
%     trainPerClass_ratio = max(20, round(numTrainSamplesPerClass * ratio));
%     fprintf('  -> 训练样本占比: %.0f%% (每类 %d 条)\n', ratio*100, trainPerClass_ratio);
    
%     for sd = 1:numSeeds
%         rng(100 + sd);  % 确保可重复
%         [Xtr, Ytr] = generateBalancedRadarData(numClasses, trainPerClass_ratio, numFeatures, T);
%         [Xte, Yte] = generateBalancedRadarData(numClasses, numTestSamplesPerClass, numFeatures, T);
        
%         % 序列 & 静态格式
%         Xtr_seq = cell(size(Xtr, 3), 1); Xte_seq = cell(size(Xte, 3), 1);
%         for i = 1:size(Xtr, 3), Xtr_seq{i} = Xtr(:, :, i)'; end
%         for i = 1:size(Xte, 3), Xte_seq{i} = Xte(:, :, i)'; end
        
%         % 仅评估代表性三种模型（时序/RNN、对比基线、本文方法）
%         for m = 1:numel(sens_models)
%             mdl_name = sens_models{m};
%             switch mdl_name
%                 case 'LSTM'
%                     layers_sens = [sequenceInputLayer(numFeatures,'Normalization','zscore'), lstmLayer(100,'OutputMode','last'), fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];
%                     opts_sens = trainingOptions('adam', 'MaxEpochs', 120, 'MiniBatchSize', 64, 'InitialLearnRate', 0.002, ...
%                         'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',40, ...
%                         'ValidationData',{Xte_seq, Yte}, 'ValidationFrequency', 25, 'Verbose', false);
%                     net_sens = trainNetwork(Xtr_seq, Ytr, layers_sens, opts_sens);
%                     Y_pred = classify(net_sens, Xte_seq);
                    
%                 case 'MPT-SFANet'
%                     layers_sens = defineMPT_SFANet_1D(numFeatures, numClasses);
%                     opts_sens = trainingOptions('adam', 'MaxEpochs', 130, 'MiniBatchSize', 64, 'InitialLearnRate', 0.001, ...
%                         'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',40, ...
%                         'ValidationData',{Xte_seq, Yte}, 'ValidationFrequency', 25, 'Verbose', false);
%                     net_sens = trainNetwork(Xtr_seq, Ytr, layers_sens, opts_sens);
%                     Y_pred = classify(net_sens, Xte_seq);
                    
%                 case 'LFT-Net'
%                     layers_sens = defineLFTNet(numFeatures, numClasses);
%                     opts_sens = trainingOptions('adam', 'MaxEpochs', 130, 'MiniBatchSize', 64, 'InitialLearnRate', 0.001, ...
%                         'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',40, ...
%                         'ValidationData',{Xte_seq, Yte}, 'ValidationFrequency', 25, 'Verbose', false);
%                     net_sens = trainNetwork(Xtr_seq, Ytr, layers_sens, opts_sens);
%                     Y_pred = classify(net_sens, Xte_seq);
%             end
            
%             [acc_tmp, ~, ~, f1_tmp] = calculateAllMetrics(Y_pred, Yte);
%             acc_stack(m, r, sd) = acc_tmp;
%             f1_stack(m, r, sd) = f1_tmp;
%             fprintf('    [%s] Seed %d -> Acc %.2f%% | F1 %.2f%%\n', mdl_name, sd, acc_tmp*100, f1_tmp*100);
%         end
%     end
% end

% acc_mean = mean(acc_stack, 3); acc_std = std(acc_stack, 0, 3);
% f1_mean  = mean(f1_stack, 3); f1_std  = std(f1_stack, 0, 3);

% % 可视化：数据量 vs 性能（均值±方差）
% figure('Position', [100, 100, 1200, 500], 'Name', 'Data-Efficiency Analysis');
% subplot(1,2,1);
% hold on; grid on;
% for m = 1:numel(sens_models)
%     errorbar(sample_ratios*100, acc_mean(m,:)*100, acc_std(m,:)*100, '-o', 'LineWidth', 1.5);
% end
% xlabel('训练样本占比 (%)'); ylabel('Accuracy (%)');
% title('数据量敏感性 - Accuracy (均值±std)');
% legend(sens_models, 'Location', 'southeast');

% subplot(1,2,2);
% hold on; grid on;
% for m = 1:numel(sens_models)
%     errorbar(sample_ratios*100, f1_mean(m,:)*100, f1_std(m,:)*100, '-o', 'LineWidth', 1.5);
% end
% xlabel('训练样本占比 (%)'); ylabel('Macro F1 (%)');
% title('数据量敏感性 - Macro F1 (均值±std)');
% legend(sens_models, 'Location', 'southeast');

%% 快速版数据量敏感性实验（4.8，快速测试，仅作验证用）
fprintf('\n======================================\n');
fprintf('  【步骤 4.8 - 快速测试】数据量敏感性 (少样本/数据效率)\n');
fprintf('======================================\n');

sample_ratios_fast = [0.30, 0.60, 1.00];  % 训练样本占比（快速）
numSeeds_fast = 1;                        % 仅做 1 次重复
sens_models_fast = {'LSTM', 'MPT-SFANet', 'LFT-Net'};

acc_stack = zeros(numel(sens_models_fast), numel(sample_ratios_fast), numSeeds_fast);
f1_stack = zeros(numel(sens_models_fast), numel(sample_ratios_fast), numSeeds_fast);

for r = 1:numel(sample_ratios_fast)
    ratio = sample_ratios_fast(r);
    trainPerClass_ratio = max(20, round(numTrainSamplesPerClass * ratio));
    fprintf('  -> 训练样本占比: %.0f%% (每类 %d 条)\n', ratio*100, trainPerClass_ratio);
    
    for sd = 1:numSeeds_fast
        rng(200 + sd);  % 固定随机种子
        [Xtr, Ytr] = generateBalancedRadarData(numClasses, trainPerClass_ratio, numFeatures, T);
        [Xte, Yte] = generateBalancedRadarData(numClasses, numTestSamplesPerClass, numFeatures, T);
        
        % 序列格式
        Xtr_seq = cell(size(Xtr, 3), 1); Xte_seq = cell(size(Xte, 3), 1);
        for i = 1:size(Xtr, 3), Xtr_seq{i} = Xtr(:, :, i)'; end
        for i = 1:size(Xte, 3), Xte_seq{i} = Xte(:, :, i)'; end
        
        for m = 1:numel(sens_models_fast)
            mdl_name = sens_models_fast{m};
            switch mdl_name
                case 'LSTM'
                    layers_sens = [
                        sequenceInputLayer(numFeatures,'Normalization','zscore')
                        lstmLayer(64,'OutputMode','last')
                        fullyConnectedLayer(numClasses)
                        softmaxLayer
                        classificationLayer];
                    opts_sens = trainingOptions('adam', 'MaxEpochs', 40, 'MiniBatchSize', 64, 'InitialLearnRate', 0.002, ...
                        'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',20, ...
                        'ValidationData',{Xte_seq, Yte}, 'ValidationFrequency', 15, 'Verbose', false);
                    net_sens = trainNetwork(Xtr_seq, Ytr, layers_sens, opts_sens);
                    Y_pred = classify(net_sens, Xte_seq);
                    
                case 'MPT-SFANet'
                    layers_sens = defineMPT_SFANet_1D(numFeatures, numClasses);
                    opts_sens = trainingOptions('adam', 'MaxEpochs', 50, 'MiniBatchSize', 64, 'InitialLearnRate', 0.001, ...
                        'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',25, ...
                        'ValidationData',{Xte_seq, Yte}, 'ValidationFrequency', 15, 'Verbose', false);
                    net_sens = trainNetwork(Xtr_seq, Ytr, layers_sens, opts_sens);
                    Y_pred = classify(net_sens, Xte_seq);
                    
                case 'LFT-Net'
                    layers_sens = defineLFTNet(numFeatures, numClasses);
                    opts_sens = trainingOptions('adam', 'MaxEpochs', 50, 'MiniBatchSize', 64, 'InitialLearnRate', 0.001, ...
                        'LearnRateSchedule','piecewise','LearnRateDropFactor',0.5,'LearnRateDropPeriod',25, ...
                        'ValidationData',{Xte_seq, Yte}, 'ValidationFrequency', 15, 'Verbose', false);
                    net_sens = trainNetwork(Xtr_seq, Ytr, layers_sens, opts_sens);
                    Y_pred = classify(net_sens, Xte_seq);
            end
            
            [acc_tmp, ~, ~, f1_tmp] = calculateAllMetrics(Y_pred, Yte);
            acc_stack(m, r, sd) = acc_tmp;
            f1_stack(m, r, sd) = f1_tmp;
            fprintf('    [%s] Seed %d -> Acc %.2f%% | F1 %.2f%%\n', mdl_name, sd, acc_tmp*100, f1_tmp*100);
        end
    end
end

acc_mean = mean(acc_stack, 3); acc_std = std(acc_stack, 0, 3);
f1_mean  = mean(f1_stack, 3); f1_std  = std(f1_stack, 0, 3);

figure('Position', [100, 100, 1200, 500], 'Name', 'Data-Efficiency Analysis (Fast)');
subplot(1,2,1);
hold on; grid on;
for m = 1:numel(sens_models_fast)
    errorbar(sample_ratios_fast*100, acc_mean(m,:)*100, acc_std(m,:)*100, '-o', 'LineWidth', 1.5);
end
xlabel('训练样本占比 (%)'); ylabel('Accuracy (%)');
title('数据量敏感性 - Accuracy');
legend(sens_models_fast, 'Location', 'southeast');

subplot(1,2,2);
hold on; grid on;
for m = 1:numel(sens_models_fast)
    errorbar(sample_ratios_fast*100, f1_mean(m,:)*100, f1_std(m,:)*100, '-o', 'LineWidth', 1.5);
end
xlabel('训练样本占比 (%)'); ylabel('Macro F1 (%)');
title('数据量敏感性 - Macro F1');
legend(sens_models_fast, 'Location', 'southeast');

%% 4.9 鲁棒性 & 校准度 & PR/ROC
fprintf('\n======================================\n');
fprintf('  【步驟 4.9】鲁棒性/校准/PR-ROC\n');
fprintf('======================================\n');

% 4.9.1 噪声鲁棒性 (仅测 MPT-SFANet 与 LFT-Net，测试集加噪)
noise_levels = [0, 0.05, 0.10, 0.20, 0.30];
acc_robust_mpt = zeros(size(noise_levels));
acc_robust_lft = zeros(size(noise_levels));

for nl = 1:numel(noise_levels)
    sig = noise_levels(nl);
    X_noisy = cellfun(@(x) max(0, min(1, x + sig*randn(size(x)))), X_test_seq, 'UniformOutput', false);
    Yp_mpt = classify(net_mpt, X_noisy);
    Yp_lft = classify(net_lft, X_noisy);
    acc_robust_mpt(nl) = mean(Yp_mpt == Y_test);
    acc_robust_lft(nl) = mean(Yp_lft == Y_test);
end

% 4.9.2 校准度 (Reliability/ECE/Brier，基于 LFT-Net 概率)
% LFT-Net 是 SeriesNetwork，使用 classify 获取概率
[~, score_lft] = classify(net_lft, X_test_seq);
classList_lft = net_lft.Layers(end).Classes; % align score columns with network class order
[~, pred_idx] = max(score_lft, [], 2);
conf = max(score_lft, [], 2);
[~, true_idx] = ismember(Y_test, classList_lft);

numBins = 10;
edges = linspace(0,1,numBins+1);
bin_conf = zeros(numBins,1); bin_acc = zeros(numBins,1); bin_count = zeros(numBins,1);
for b = 1:numBins
    mask = conf > edges(b) & conf <= edges(b+1);
    bin_count(b) = sum(mask);
    if bin_count(b) > 0
        bin_conf(b) = mean(conf(mask));
        bin_acc(b) = mean(pred_idx(mask) == true_idx(mask));
    end
end
ece = nansum(abs(bin_conf - bin_acc) .* (bin_count / numel(conf)));

% Brier Score (多分类)
nSamples = numel(Y_test);
numC = numel(classList_lft);
onehot = zeros(nSamples, numC);
for i = 1:nSamples
    onehot(i, true_idx(i)) = 1;
end
brier = mean(sum((score_lft - onehot).^2, 2));

fprintf('  校准指标: ECE = %.4f, Brier = %.4f\n', ece, brier);

% 4.9.3 PR/ROC (关键类：C2=Bird, C8=UAV) 对比 MPT vs LFT
keyClasses = [2, 8];
pr_curves = cell(numel(keyClasses),2); roc_curves = cell(numel(keyClasses),2); auc_store = zeros(numel(keyClasses),2);
% MPT-SFANet 是 SeriesNetwork，使用 classify 获取概率
[~, score_mpt] = classify(net_mpt, X_test_seq);
yt_numeric = true_idx;

for k = 1:numel(keyClasses)
    cls = keyClasses(k);
    target = (yt_numeric == cls);
    % LFT
    [rec_lft, prec_lft, ~, aucpr_lft] = perfcurve(target, score_lft(:,cls), true, 'xCrit','reca','yCrit','prec');
    [fpr_lft, tpr_lft, ~, aucroc_lft] = perfcurve(target, score_lft(:,cls), true);
    pr_curves{k,1} = {rec_lft, prec_lft, aucpr_lft};
    roc_curves{k,1} = {fpr_lft, tpr_lft, aucroc_lft};
    % MPT
    [rec_mpt, prec_mpt, ~, aucpr_mpt] = perfcurve(target, score_mpt(:,cls), true, 'xCrit','reca','yCrit','prec');
    [fpr_mpt, tpr_mpt, ~, aucroc_mpt] = perfcurve(target, score_mpt(:,cls), true);
    pr_curves{k,2} = {rec_mpt, prec_mpt, aucpr_mpt};
    roc_curves{k,2} = {fpr_mpt, tpr_mpt, aucroc_mpt};
    auc_store(k, :) = [aucroc_lft, aucroc_mpt];
end

% 可视化鲁棒性/校准/PR/ROC
figure('Position', [80, 80, 1600, 900], 'Name', 'Robustness & Calibration & PR-ROC');
% 噪声鲁棒性
subplot(2,2,1);
plot(noise_levels*100, acc_robust_lft*100, '-o','LineWidth',1.5); hold on;
plot(noise_levels*100, acc_robust_mpt*100, '-s','LineWidth',1.5);
grid on; xlabel('测试噪声幅度 (%)'); ylabel('Accuracy (%)');
title('噪声鲁棒性 (测试集加噪)'); legend({'LFT-Net','MPT-SFANet'},'Location','southwest');

% 校准曲线
subplot(2,2,2);
plot([0 1],[0 1],'k--'); hold on; grid on;
plot(bin_conf, bin_acc, '-o','LineWidth',1.5);
xlabel('平均置信度'); ylabel('经验准确率');
title(sprintf('可靠性曲线 (ECE=%.4f, Brier=%.4f)', ece, brier));

% PR 曲线 (C2/C8)
subplot(2,2,3);
colors = lines(2);
for k = 1:numel(keyClasses)
    rec_l = pr_curves{k,1}{1}; prec_l = pr_curves{k,1}{2}; aucpr_l = pr_curves{k,1}{3};
    rec_m = pr_curves{k,2}{1}; prec_m = pr_curves{k,2}{2}; aucpr_m = pr_curves{k,2}{3};
    plot(rec_l, prec_l, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', sprintf('LFT C%d AUPR %.3f', keyClasses(k), aucpr_l)); hold on;
    plot(rec_m, prec_m, '--', 'Color', colors(k,:), 'LineWidth', 1.2, 'DisplayName', sprintf('MPT C%d AUPR %.3f', keyClasses(k), aucpr_m));
end
grid on; xlabel('Recall'); ylabel('Precision');
title('关键类 PR 曲线 (C2=Bird, C8=UAV)'); legend('Location','southwest');

% ROC 曲线 (C2/C8)
subplot(2,2,4);
for k = 1:numel(keyClasses)
    fpr_l = roc_curves{k,1}{1}; tpr_l = roc_curves{k,1}{2}; auc_l = roc_curves{k,1}{3};
    fpr_m = roc_curves{k,2}{1}; tpr_m = roc_curves{k,2}{2}; auc_m = roc_curves{k,2}{3};
    plot(fpr_l, tpr_l, 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', sprintf('LFT C%d AUC %.3f', keyClasses(k), auc_l)); hold on;
    plot(fpr_m, tpr_m, '--', 'Color', colors(k,:), 'LineWidth', 1.2, 'DisplayName', sprintf('MPT C%d AUC %.3f', keyClasses(k), auc_m));
end
grid on; xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('关键类 ROC 曲线'); legend('Location','southeast');

%% 5. 绘图：混淆矩阵对比 (定性分析)
figure( 'Name', 'Comparison of Confusion Matrices');
for i = 1:6
    subplot(2, 3, i);
    cm_chart = confusionchart(Y_test, preds_list{i});
    cm_chart.Title = sprintf('%s\nF1-Score: %.2f%%', models_name{i}, results_matrix(i, 4)*100);
    cm_chart.RowSummary = 'row-normalized'; % 显示召回率
    cm_chart.FontSize = 10;
    
    % 高亮对比文献和本文方法
end
sgtitle('Confusion Matrix Comparison: Baselines vs. MPT-SFANet vs. LFT-Net (Ours)', 'FontSize', 16, 'FontWeight', 'bold');

%% 6. 绘图：多维度指标雷达图 (可选，直观展示)
figure( 'Name', 'Metrics Bar Chart');
b = bar(results_matrix * 100);
xticklabels(models_name);
ylabel('Score (%)');
legend({'Accuracy', 'Precision', 'Recall', 'F1-Score'}, 'Location', 'southeast');
grid on;
ylim([min(results_matrix(:))*100 - 5, 100]);
title('Comprehensive Performance Comparison');
% 高亮 Ours
b(4).FaceColor = [0.8500 0.3250 0.0980]; % F1-Score 用醒目颜色

%% 辅助函数：全指标计算
function [acc, macro_prec, macro_rec, macro_f1] = calculateAllMetrics(Y_pred, Y_true)
    % 获取所有类别
    classes = categories(Y_true);
    numClasses = length(classes);
    
    % 计算混淆矩阵
    cm = confusionmat(Y_true, Y_pred);
    
    % 1. Accuracy
    acc = sum(diag(cm)) / sum(cm(:));
    
    % 初始化每类的指标
    precisions = zeros(numClasses, 1);
    recalls = zeros(numClasses, 1);
    f1s = zeros(numClasses, 1);
    
    for i = 1:numClasses
        tp = cm(i,i);
        fp = sum(cm(:,i)) - tp;
        fn = sum(cm(i,:)) - tp;
        
        % Precision = TP / (TP + FP)
        if (tp + fp) > 0
            precisions(i) = tp / (tp + fp);
        else
            precisions(i) = 0;
        end
        
        % Recall = TP / (TP + FN)
        if (tp + fn) > 0
            recalls(i) = tp / (tp + fn);
        else
            recalls(i) = 0;
        end
        
        % F1 = 2 * P * R / (P + R)
        if (precisions(i) + recalls(i)) > 0
            f1s(i) = 2 * (precisions(i) * recalls(i)) / (precisions(i) + recalls(i));
        else
            f1s(i) = 0;
        end
    end
    
    % Macro-Average (多分类任务的标准做法)
    macro_prec = mean(precisions);
    macro_rec = mean(recalls);
    macro_f1 = mean(f1s);
end

function updateProgress(fig, current, total, name)
    figure(fig);
    clf;
    progress = current / total;
    barh(progress, 'FaceColor', [0.2 0.6 1]);
    xlim([0 1]);
    text(0.1, 1, sprintf('Training: %s (%d/%d)', name, current, total), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Progress');
    axis off;
    drawnow;
end

function label = scores2label(scores, classes)
    % 将网络输出的分数矩阵转换为分类标签（与 Old.m 中实现一致）
    [~, idx] = max(scores, [], 2);
    label = categorical(classes(idx));
end

function scores = predict_with_format(model, data)
    % 辅助函数：处理 dlnetwork 的预测
    % dlnetwork 从 trainnet 返回，minibatchpredict 可以直接使用
    if isa(model, 'dlnetwork')
        % dlnetwork：使用 minibatchpredict
        % 注意：minibatchpredict 可以直接处理 [numSamples x numFeatures] 或 cell 数据
        scores = minibatchpredict(model, data);
        % 确保输出是普通数组而不是 dlarray
        if isdlarray(scores)
            scores = extractdata(scores);
        end
    else
        % SeriesNetwork：直接使用 classify
        [~, scores] = classify(model, data);
    end
end
