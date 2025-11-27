% 快速测试脚本 - 验证 trainnet + minibatchpredict 的修复
clear; clc; rng(42);

fprintf('=== 快速API测试 ===\n\n');

% 生成小规模测试数据
numClasses = 3;
numFeatures = 10;
numSamples = 30;
T = 10;

% 静态数据（用于MLP）
X_static_train = randn(numSamples, numFeatures);
X_static_test = randn(5, numFeatures);

% 序列数据（用于LSTM/GRU）
X_seq_train = cell(numSamples, 1);
X_seq_test = cell(5, 1);
for i = 1:numSamples
    X_seq_train{i} = randn(numFeatures, T)';
end
for i = 1:5
    X_seq_test{i} = randn(numFeatures, T)';
end

Y_train = categorical(randi(numClasses, numSamples, 1));
Y_test = categorical(randi(numClasses, 5, 1));

fprintf('数据准备完成。\n');
fprintf('  - 静态数据: [%d x %d]\n', numSamples, numFeatures);
fprintf('  - 序列数据: %d samples x (%d x %d)\n\n', numSamples, T, numFeatures);

% 测试 1: MLP with trainnet
fprintf('【测试1】MLP with trainnet...\n');
layers_mlp = [
    featureInputLayer(numFeatures, 'Normalization', 'zscore')
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numClasses)
];
opts_mlp = trainingOptions('adam', 'MaxEpochs', 5, 'MiniBatchSize', 16, 'Verbose', false);

try
    [net_mlp, info] = trainnet(X_static_train, Y_train, layers_mlp, 'crossentropy', opts_mlp);
    fprintf('  ✓ trainnet 成功\n');
    
    % 测试预测
    scores_mlp = predict_with_format(net_mlp, X_static_test);
    fprintf('  ✓ predict_with_format 成功，输出大小: [%d x %d]\n', size(scores_mlp, 1), size(scores_mlp, 2));
    
    % 转换为标签
    pred_mlp = scores2label(scores_mlp, categories(Y_train));
    fprintf('  ✓ scores2label 成功\n\n');
    
catch ME
    fprintf('  ✗ 错误: %s\n\n', ME.message);
end

% 测试 2: LSTM with trainnet
fprintf('【测试2】LSTM with trainnet...\n');
layers_lstm = [
    sequenceInputLayer(numFeatures, 'Normalization', 'zscore')
    lstmLayer(32, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
];
opts_lstm = trainingOptions('adam', 'MaxEpochs', 5, 'MiniBatchSize', 8, 'Verbose', false);

try
    [net_lstm, info] = trainnet(X_seq_train, Y_train, layers_lstm, 'crossentropy', opts_lstm);
    fprintf('  ✓ trainnet 成功\n');
    
    % 测试预测
    scores_lstm = predict_with_format(net_lstm, X_seq_test);
    fprintf('  ✓ predict_with_format 成功，输出大小: [%d x %d]\n', size(scores_lstm, 1), size(scores_lstm, 2));
    
    % 转换为标签
    pred_lstm = scores2label(scores_lstm, categories(Y_train));
    fprintf('  ✓ scores2label 成功\n\n');
    
catch ME
    fprintf('  ✗ 错误: %s\n\n', ME.message);
end

% 测试 3: 验证 dlnetwork 检测
fprintf('【测试3】验证网络类型检测...\n');
fprintf('  MLP 网络类型: %s\n', class(net_mlp));
fprintf('  LSTM 网络类型: %s\n', class(net_lstm));
fprintf('  是否是 dlnetwork: MLP=%d, LSTM=%d\n\n', isa(net_mlp, 'dlnetwork'), isa(net_lstm, 'dlnetwork'));

fprintf('✅ 所有测试完成！\n');

% ===== 辅助函数 =====
function scores = predict_with_format(model, data)
    if isa(model, 'dlnetwork')
        if iscell(data)
            nSamples = numel(data);
            
            % 第一次预测以确定输出大小
            sample_dlarray = dlarray(data{1}, 'TCB');
            batch_scores = minibatchpredict(model, sample_dlarray);
            batch_scores_extracted = extractdata(batch_scores);
            numOutputs = size(batch_scores_extracted, 2);
            
            % 预分配输出
            scores = zeros(nSamples, numOutputs, 'single');
            scores(1, :) = batch_scores_extracted;
            
            % 后续样本预测
            for i = 2:nSamples
                sample_dlarray = dlarray(data{i}, 'TCB');
                batch_scores = minibatchpredict(model, sample_dlarray);
                scores(i, :) = extractdata(batch_scores);
            end
        else
            if isrow(data)
                data_dlarray = dlarray(data', 'CB');
            else
                data_dlarray = dlarray(data, 'CB');
            end
            batch_scores = minibatchpredict(model, data_dlarray);
            scores = extractdata(batch_scores);
        end
    else
        [~, scores] = classify(model, data);
    end
end

function label = scores2label(scores, classes)
    [~, idx] = max(scores, [], 2);
    label = categorical(classes(idx));
end
