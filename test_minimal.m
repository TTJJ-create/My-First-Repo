% 极简测试 - trainnet 基础验证
clear; clc; rng(42);

fprintf('=== trainnet 基础测试 ===\n\n');

% 生成最小数据
fprintf('【1】生成数据\n');
X_train = randn(20, 5);  % 20 samples, 5 features
temp = repmat([1 2 3], 1, ceil(20/3));
Y_train = categorical(temp(1:20))';  % 20 labels
fprintf('  X_train: [%d x %d]\n', size(X_train, 1), size(X_train, 2));
fprintf('  Y_train: [%d x 1]\n', numel(Y_train));

fprintf('\n【2】定义网络\n');
layers = [
    featureInputLayer(5, 'Name', 'input', 'Normalization', 'zscore')
    fullyConnectedLayer(16, 'Name', 'fc1')
    reluLayer('Name', 'relu')
    fullyConnectedLayer(3, 'Name', 'fc_out')
];
fprintf('  网络定义完成\n');

fprintf('\n【3】训练\n');
opts = trainingOptions('adam', ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 8, ...
    'Verbose', false);

try
    tic;
    [net, info] = trainnet(X_train, Y_train, layers, 'crossentropy', opts);
    t = toc;
    fprintf('  ✓ trainnet 成功 (%.1fs)\n', t);
    fprintf('  网络类型: %s\n', class(net));
    fprintf('  是否 dlnetwork: %d\n', isa(net, 'dlnetwork'));
catch ME
    fprintf('  ✗ 训练失败: %s\n', ME.message);
    return;
end

fprintf('\n【4】预测\n');
X_test = randn(5, 5);  % 5 samples, 5 features
fprintf('  X_test: [%d x %d]\n', size(X_test, 1), size(X_test, 2));

try
    % 直接调用 minibatchpredict
    fprintf('  - 尝试 minibatchpredict...\n');
    scores = minibatchpredict(net, X_test);
    fprintf('  ✓ minibatchpredict 成功\n');
    fprintf('  输出大小: [%d x %d]\n', size(scores, 1), size(scores, 2));
    fprintf('  输出类型: %s\n', class(scores));
    
    if isdlarray(scores)
        fprintf('  - 是 dlarray，正在提取...\n');
        scores = extractdata(scores);
        fprintf('  提取后大小: [%d x %d]\n', size(scores, 1), size(scores, 2));
    end
    
    % 获取预测标签
    [~, idx] = max(scores, [], 2);
    labels = categorical(idx);
    fprintf('  ✓ 预测成功，标签: %d\n', numel(labels));
    
catch ME
    fprintf('  ✗ 预测失败: %s\n', ME.message);
    fprintf('  标识符: %s\n', ME.identifier);
    return;
end

fprintf('\n✅ 所有测试通过!\n');
