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
fig = figure('Position', [100, 100, 1000, 600], 'Name', 'Training Monitor');

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
svm_mdl = fitcecoc(X_train_static, Y_train, 'Learners', templateSVM('KernelFunction','linear'));
Y_pred_svm = predict(svm_mdl, X_test_static);

% 2. MLP
fprintf('=== 2/6 训练 MLP (NN Baseline) ===\n');
updateProgress(fig, 2, 6, 'MLP');
layers_mlp = [featureInputLayer(numFeatures,'Normalization','zscore'), fullyConnectedLayer(128), reluLayer, fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];
net_mlp = trainNetwork(X_train_static, Y_train, layers_mlp, trainingOptions('adam', 'MaxEpochs', 100, 'Verbose', false));
Y_pred_mlp = classify(net_mlp, X_test_static);

% 3. LSTM
fprintf('=== 3/6 训练 LSTM (RNN Baseline) ===\n');
updateProgress(fig, 3, 6, 'LSTM');
layers_lstm = [sequenceInputLayer(numFeatures,'Normalization','zscore'), lstmLayer(100,'OutputMode','last'), fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];
net_lstm = trainNetwork(X_train_seq, Y_train, layers_lstm, getOpts(0.002));
Y_pred_lstm = classify(net_lstm, X_test_seq);

% 4. GRU
fprintf('=== 4/6 训练 GRU (RNN Baseline) ===\n');
updateProgress(fig, 4, 6, 'GRU');
layers_gru = [sequenceInputLayer(numFeatures,'Normalization','zscore'), gruLayer(100,'OutputMode','last'), fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];
net_gru = trainNetwork(X_train_seq, Y_train, layers_gru, getOpts(0.002));
Y_pred_gru = classify(net_gru, X_test_seq);

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
close(fig); % 关闭训练监控窗口
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
t_mlp = measure_time(net_mlp, test_sample_static, 'classify');

% 3. LSTM 推理时间
fprintf('  3. 测量 LSTM...\n');
t_lstm = measure_time(net_lstm, test_sample_seq, 'classify');

% 4. GRU 推理时间
fprintf('  4. 测量 GRU...\n');
t_gru = measure_time(net_gru, test_sample_seq, 'classify');

% 5. MPT-SFANet 推理时间
fprintf('  5. 测量 MPT-SFANet...\n');
t_mpt = measure_time(net_mpt, test_sample_seq, 'classify');

% 6. LFT-Net 推理时间
fprintf('  6. 测量 LFT-Net...\n');
t_lft = measure_time(net_lft, test_sample_seq, 'classify');

% 汇总时间数据
times_list = [t_svm, t_mlp, t_lstm, t_gru, t_mpt, t_lft];
fprintf('  测量完成。\n');

%% 5. 绘图：混淆矩阵对比 (定性分析)
figure('Position', [50, 50, 1600, 900], 'Name', 'Comparison of Confusion Matrices');
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
% 如果没有 radarplot 函数，可以用柱状图替代
figure('Position', [100, 100, 800, 500], 'Name', 'Metrics Bar Chart');
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