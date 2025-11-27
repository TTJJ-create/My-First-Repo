% 测试 trainnet 的具体错误
clear; clc;

rng(42);
numClasses = 9;
numFeatures = 10;
T = 10;

% 生成简单测试数据
X_test = randn(numFeatures, 1);
Y_test = categorical([1; 2; 3]);
X_train = randn(numFeatures, 3);
Y_train = categorical([1; 2; 3]);

% MLP 网络定义
layers_mlp = [
    featureInputLayer(numFeatures, 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(3, 'Name', 'fc_out')
];

% 训练选项
opts = trainingOptions('adam', ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 1, ...
    'Verbose', true);

try
    fprintf('正在用 trainnet 训练 MLP...\n');
    [net, info] = trainnet(X_train, Y_train, layers_mlp, 'crossentropy', opts);
    fprintf('trainnet 成功\n');
    
    fprintf('正在用 minibatchpredict 预测...\n');
    Y_scores = minibatchpredict(net, X_test);
    fprintf('minibatchpredict 成功\n');
    fprintf('输出大小: %s\n', mat2str(size(Y_scores)));
    
catch ME
    fprintf('错误: %s\n', ME.message);
    fprintf('标识符: %s\n', ME.identifier);
end
