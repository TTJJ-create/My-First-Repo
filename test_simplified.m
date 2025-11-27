%% 简化测试 - trainnet 修复验证
% 使用较小的数据集和epoch数快速验证修复是否正确
clear; close all; clc;
rng(42);

fprintf('===== 简化测试：验证 trainnet 修复 =====\n\n');

numClasses = 3;
numFeatures = 5;
numSamples = 60;
T = 5;

% 生成测试数据
fprintf('【1】生成小规模测试数据\n');
X_static_train = randn(numSamples, numFeatures);
X_static_test = randn(20, numFeatures);
X_seq_train = cell(numSamples, 1);
X_seq_test = cell(20, 1);
for i = 1:numSamples
    X_seq_train{i} = randn(numFeatures, T)';
end
for i = 1:20
    X_seq_test{i} = randn(numFeatures, T)';
end
Y_train = categorical(randi(numClasses, numSamples, 1));
Y_test = categorical(randi(numClasses, 20, 1));
fprintf('  ✓ 完成\n\n');

% 测试1：MLP
fprintf('【2】测试 MLP (trainnet + predict_with_format)\n');
try
    layers_mlp = [
        featureInputLayer(numFeatures, 'Normalization', 'zscore')
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(numClasses)
    ];
    opts = trainingOptions('adam', 'MaxEpochs', 3, 'MiniBatchSize', 16, ...
        'Verbose', false, 'Plots', 'none');
    [net_mlp, ~] = trainnet(X_static_train, Y_train, layers_mlp, 'crossentropy', opts);
    fprintf('  ✓ trainnet 训练成功\n');
    
    % 测试预测
    Y_scores_mlp = predict_with_format(net_mlp, X_static_test);
    fprintf('  ✓ predict_with_format 输出: [%d x %d]\n', size(Y_scores_mlp, 1), size(Y_scores_mlp, 2));
    
    % 转换标签
    Y_pred_mlp = scores2label(Y_scores_mlp, categories(Y_train));
    fprintf('  ✓ scores2label 转换成功\n');
    
    % 计算准确率
    acc_mlp = mean(Y_pred_mlp == Y_test);
    fprintf('  ✓ 准确率: %.2f%%\n\n', acc_mlp * 100);
    
catch ME
    fprintf('  ✗ 错误: %s\n\n', ME.message);
    return;
end

% 测试2：LSTM
fprintf('【3】测试 LSTM (trainnet + predict_with_format)\n');
try
    layers_lstm = [
        sequenceInputLayer(numFeatures, 'Normalization', 'zscore')
        lstmLayer(16, 'OutputMode', 'last')
        fullyConnectedLayer(numClasses)
    ];
    opts = trainingOptions('adam', 'MaxEpochs', 3, 'MiniBatchSize', 8, ...
        'Verbose', false, 'Plots', 'none');
    [net_lstm, ~] = trainnet(X_seq_train, Y_train, layers_lstm, 'crossentropy', opts);
    fprintf('  ✓ trainnet 训练成功\n');
    
    % 测试预测
    Y_scores_lstm = predict_with_format(net_lstm, X_seq_test);
    fprintf('  ✓ predict_with_format 输出: [%d x %d]\n', size(Y_scores_lstm, 1), size(Y_scores_lstm, 2));
    
    % 转换标签
    Y_pred_lstm = scores2label(Y_scores_lstm, categories(Y_train));
    fprintf('  ✓ scores2label 转换成功\n');
    
    % 计算准确率
    acc_lstm = mean(Y_pred_lstm == Y_test);
    fprintf('  ✓ 准确率: %.2f%%\n\n', acc_lstm * 100);
    
catch ME
    fprintf('  ✗ 错误: %s\n\n', ME.message);
    return;
end

% 测试3：SeriesNetwork (trainNetwork) 对比
fprintf('【4】测试 SeriesNetwork (trainNetwork - 参考)\n');
try
    layers_net = [
        sequenceInputLayer(numFeatures, 'Normalization', 'zscore')
        lstmLayer(16, 'OutputMode', 'last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];
    opts = trainingOptions('adam', 'MaxEpochs', 3, 'MiniBatchSize', 8, ...
        'Verbose', false, 'Plots', 'none');
    net_series = trainNetwork(X_seq_train, Y_train, layers_net, opts);
    fprintf('  ✓ trainNetwork 训练成功\n');
    
    % 测试预测 (SeriesNetwork)
    [Y_pred_series, Y_scores_series] = classify(net_series, X_seq_test);
    fprintf('  ✓ classify 输出: labels=[%d x 1], scores=[%d x %d]\n', ...
        numel(Y_pred_series), size(Y_scores_series, 1), size(Y_scores_series, 2));
    
    % 计算准确率
    acc_series = mean(Y_pred_series == Y_test);
    fprintf('  ✓ 准确率: %.2f%%\n\n', acc_series * 100);
    
catch ME
    fprintf('  ✗ 错误: %s\n\n', ME.message);
    return;
end

fprintf('===== ✅ 所有测试通过！=====\n');
fprintf('MLP (trainnet):    %.2f%%\n', acc_mlp * 100);
fprintf('LSTM (trainnet):   %.2f%%\n', acc_lstm * 100);
fprintf('LSTM (trainNet):   %.2f%%\n', acc_series * 100);

%% 辅助函数
function scores = predict_with_format(model, data)
    if isa(model, 'dlnetwork')
        scores = minibatchpredict(model, data);
        if isdlarray(scores)
            scores = extractdata(scores);
        end
    else
        [~, scores] = classify(model, data);
    end
end

function label = scores2label(scores, classes)
    [~, idx] = max(scores, [], 2);
    label = categorical(classes(idx));
end
