%% LFT-Net (Proposed) vs MPT-SFANet (TAES 2024 Baseline)
% 目标：验证 Proposed Method 在低维时序雷达数据上优于 Image-based SOTA 的适配版
clear; close all; clc;
rng(42);

%% 1. 参数设置
fprintf('==================================================\n');
fprintf('  LFT-Net vs MPT-SFANet 对比实验 (TAES投稿准备)\n');
fprintf('==================================================\n\n');

numClasses = 9;
numTrainSamplesPerClass = 600;
numTestSamplesPerClass = 200;
numFeatures = 10;
T = 10;

%% 2. 数据准备
fprintf('【步骤1】生成数据...\n');
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

%% 3. 模型定义与训练
fig = figure('Position', [100, 100, 1400, 800], 'Name', 'Training Monitor');

% --- 通用训练选项 ---
getOpts = @(lr) trainingOptions('adam', ...
    'MaxEpochs', 150, 'MiniBatchSize', 64, ...
    'InitialLearnRate', lr, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 40, ...
    'ValidationData', {X_test_seq, Y_test}, 'ValidationFrequency', 30, ...
    'Plots', 'training-progress', 'Verbose', false);

% 1. SVM (Baseline Static)
fprintf('=== 训练 SVM ===\n');
updateProgress(fig, 1, 6, 'SVM (Baseline)');
svm_mdl = fitcecoc(X_train_static, Y_train, 'Learners', templateSVM('KernelFunction','linear'));
Y_pred_svm = predict(svm_mdl, X_test_static);
acc_svm = mean(Y_pred_svm == Y_test);

% 2. MLP (Baseline NN)
fprintf('=== 训练 MLP ===\n');
updateProgress(fig, 2, 6, 'MLP (Baseline)');
layers_mlp = [featureInputLayer(numFeatures,'Normalization','zscore'), fullyConnectedLayer(128), reluLayer, fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];
opts_mlp = trainingOptions('adam', 'MaxEpochs', 100, 'ValidationData', {X_test_static, Y_test}, 'Verbose', false);
net_mlp = trainNetwork(X_train_static, Y_train, layers_mlp, opts_mlp);
Y_pred_mlp = classify(net_mlp, X_test_static);
acc_mlp = mean(Y_pred_mlp == Y_test);

% 3. LSTM (Baseline RNN)
fprintf('=== 训练 LSTM ===\n');
updateProgress(fig, 3, 6, 'LSTM (Temporal)');
layers_lstm = [sequenceInputLayer(numFeatures,'Normalization','zscore'), lstmLayer(100,'OutputMode','last'), fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];
net_lstm = trainNetwork(X_train_seq, Y_train, layers_lstm, getOpts(0.002));
Y_pred_lstm = classify(net_lstm, X_test_seq);
acc_lstm = mean(Y_pred_lstm == Y_test);

% 4. GRU (Baseline RNN)
fprintf('=== 训练 GRU ===\n');
updateProgress(fig, 4, 6, 'GRU (Temporal)');
layers_gru = [sequenceInputLayer(numFeatures,'Normalization','zscore'), gruLayer(100,'OutputMode','last'), fullyConnectedLayer(numClasses), softmaxLayer, classificationLayer];
net_gru = trainNetwork(X_train_seq, Y_train, layers_gru, getOpts(0.002));
Y_pred_gru = classify(net_gru, X_test_seq);
acc_gru = mean(Y_pred_gru == Y_test);

% 5. MPT-SFANet (Comparison - TAES Paper)
fprintf('=== 训练 MPT-SFANet (对比文献) ===\n');
updateProgress(fig, 5, 6, 'MPT-SFANet (Comparison)');
layers_mpt = defineMPT_SFANet_1D(numFeatures, numClasses);
net_mpt = trainNetwork(X_train_seq, Y_train, layers_mpt, getOpts(0.001));
Y_pred_mpt = classify(net_mpt, X_test_seq);
acc_mpt = mean(Y_pred_mpt == Y_test);

% 6. LFT-Net (Proposed)
fprintf('=== 训练 LFT-Net (Ours) ===\n');
updateProgress(fig, 6, 6, 'LFT-Net (Ours)');
layers_lft = defineLFTNet(numFeatures, numClasses);
net_lft = trainNetwork(X_train_seq, Y_train, layers_lft, getOpts(0.001));
Y_pred_lft = classify(net_lft, X_test_seq);
acc_lft = mean(Y_pred_lft == Y_test);

%% 4. 结果可视化与对比
fprintf('\n======================================\n');
fprintf('         实验结果汇总 (Accuracy)\n');
fprintf('======================================\n');
fprintf('SVM:        %.2f%%\n', acc_svm*100);
fprintf('MLP:        %.2f%%\n', acc_mlp*100);
fprintf('LSTM:       %.2f%%\n', acc_lstm*100);
fprintf('GRU:        %.2f%%\n', acc_gru*100);
fprintf('MPT-SFANet: %.2f%% (Contrast)\n', acc_mpt*100);
fprintf('LFT-Net:    %.2f%% (Ours)\n', acc_lft*100);
fprintf('======================================\n');

% 绘制混淆矩阵对比图
figure( 'Name', 'Comparison of Confusion Matrices');

models = {
    'SVM', Y_pred_svm, acc_svm;
    'MLP', Y_pred_mlp, acc_mlp;
    'LSTM', Y_pred_lstm, acc_lstm;
    'GRU', Y_pred_gru, acc_gru;
    'MPT-SFANet', Y_pred_mpt, acc_mpt;
    'LFT-Net', Y_pred_lft, acc_lft
};

for i = 1:6
    subplot(2, 3, i);
    cm = confusionchart(Y_test, models{i, 2}, 'XLabel', 'Predicted Classes', 'YLabel', 'True Classes');
    cm.Title = sprintf('%s: %.1f%%', models{i, 1}, models{i, 3}*100);
    % cm.RowSummary = 'row-normalized';
    cm.FontSize = 8;
    
    % 为Ours和Contrast加颜色高亮边框逻辑（视觉上区分）
if i >= 5
    % 1. 处理标题文本（兼容 cell/字符数组/字符串）
    titleText = cm.Title;
    if iscell(titleText)
        titleText = titleText{1};
    end
    % 防止文本为空，设置默认值
    if isempty(titleText)
        titleText = 'Confusion Matrix';
    end
    
    % 2. 获取混淆矩阵所在的图窗（确保是当前图窗）
    fig = cm.Parent;
    while ~isa(fig, 'matlab.ui.Figure')  % 确保找到最顶层图窗
        fig = fig.Parent;
    end
    
    % 3. 手动添加标题（annotation 函数创建独立标题对象，句柄必有效）
    % 位置：图窗顶部居中（x从0.1到0.9，y=0.95，适配大多数布局）
    titleObj = annotation(fig, 'textbox', [0.1, 0.95, 0.8, 0.05], ...
        'String', titleText, ...
        'HorizontalAlignment', 'center', ...  % 水平居中
        'VerticalAlignment', 'middle', ...    % 垂直居中
        'EdgeColor', 'none', ...              % 去掉文本框边框
        'Color', 'red', ...                   % 字体颜色红色
        'FontWeight', 'bold', ...             % 字体加粗
        'FontSize', 12);                      % 可选：调整字体大小
end
end

sgtitle('Performance Comparison including TAES Baseline', 'FontSize', 16, 'FontWeight', 'bold');

%% 5. 关键指标表格生成 (F1-Score Analysis)
% 计算飞鸟(Class 2)和无人机(Class 8)的F1分数，这对雷达识别至关重要
calc_f1 = @(pred, gt, cls) 2 * (sum(pred==cls & gt==cls) / (sum(pred==cls) + sum(gt==cls) + eps));

fprintf('\n【难分类别 F1-Score 对比】\n');
fprintf('Method      \t Bird (C2) \t UAV (C8) \t Overall Acc\n');
fprintf('----------------------------------------------------\n');
methods_list = {'SVM','MLP','LSTM','GRU','MPT-SFANet','LFT-Net'};
preds_list = {Y_pred_svm, Y_pred_mlp, Y_pred_lstm, Y_pred_gru, Y_pred_mpt, Y_pred_lft};
accs_list = [acc_svm, acc_mlp, acc_lstm, acc_gru, acc_mpt, acc_lft];

for i = 1:6
    f1_bird = calc_f1(preds_list{i}, Y_test, categorical(2));
    f1_uav = calc_f1(preds_list{i}, Y_test, categorical(8));
    fprintf('%-12s\t %.1f%%\t\t %.1f%%\t\t %.1f%%\n', ...
        methods_list{i}, f1_bird*100, f1_uav*100, accs_list(i)*100);
end
fprintf('----------------------------------------------------\n');

%% 辅助函数
function updateProgress(fig, current, total, name)
    figure(fig);
    clf;
    progress = current / total;
    barh(progress, 'FaceColor', [0.2 0.7 0.3]);
    xlim([0 1]);
    title(sprintf('Training Progress: %d/%d - %s', current, total, name), 'FontSize', 14);
    xlabel('Completion');
    drawnow;
end