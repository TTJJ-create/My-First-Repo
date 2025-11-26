function lgraph = defineLFTNet(numFeatures, numClasses)
% LFT-Net增强版 - 适应困难数据
% 改进：
% 1. 增加模型容量 (embedDim 32→48)
% 2. 增强正则化 (dropout 0.3→0.4)
% 3. 保持2层编码器（平衡性能和复杂度）

fprintf('构建增强版 LFT-Net (最大化性能)...\n');

% 大幅增加容量
% embedDim = 80;      % 48→64 (增强)
% numHeads = 4;       % 保持
% ffnDim = 160;       % 96→128 (增强)
% numEncoders = 3;    % 2→3 (增加层数)
% dropout = 0.18;     % 0.35→0.25 (降低正则化)

embedDim = 64;      % 从 80 降回 64 (特征更紧凑，减少过拟合)
numHeads = 4;       % 每个头 16 维，刚好
ffnDim = 256;       % 从 160 提升到 256 (4倍膨胀！这是提分的关键)
numEncoders = 3;    % 保持 3 层
dropout = 0.25;     % 稍微回调到 0.25，配合大的 FFN 防止死记硬背

lgraph = layerGraph();

%% 1. 输入层
inputLayer = sequenceInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph, inputLayer);

%% 2. 特征解耦模块
staticBranch = [
    fullyConnectedLayer(embedDim/2, 'Name', 'static_proj')
    layerNormalizationLayer('Name', 'static_norm')
    reluLayer('Name', 'static_relu')
    dropoutLayer(0.3, 'Name', 'static_dropout')
];

dynamicBranch = [
    fullyConnectedLayer(embedDim/2, 'Name', 'dynamic_proj')
    layerNormalizationLayer('Name', 'dynamic_norm')
    reluLayer('Name', 'dynamic_relu')
    dropoutLayer(0.3, 'Name', 'dynamic_dropout')
];

lgraph = addLayers(lgraph, staticBranch);
lgraph = addLayers(lgraph, dynamicBranch);
lgraph = connectLayers(lgraph, 'input', 'static_proj');
lgraph = connectLayers(lgraph, 'input', 'dynamic_proj');

concatLayer = concatenationLayer(1, 2, 'Name', 'feature_concat');
lgraph = addLayers(lgraph, concatLayer);
lgraph = connectLayers(lgraph, 'static_dropout', 'feature_concat/in1');
lgraph = connectLayers(lgraph, 'dynamic_dropout', 'feature_concat/in2');

%% 3. 特征嵌入
embedLayers = [
    fullyConnectedLayer(embedDim, 'Name', 'embed_proj')
    layerNormalizationLayer('Name', 'embed_norm')
    dropoutLayer(0.2, 'Name', 'embed_drop')
];
lgraph = addLayers(lgraph, embedLayers);
lgraph = connectLayers(lgraph, 'feature_concat', 'embed_proj');

%% 4. Transformer编码器（2层）
currentLayer = 'embed_drop';
for i = 1:numEncoders
    lgraph = addTransformerBlock(lgraph, currentLayer, i, numHeads, embedDim, ffnDim, dropout);
    currentLayer = sprintf('enc%d_add2', i);
end

%% 5. 多尺度池化
finalLN = layerNormalizationLayer('Name', 'final_ln');
gap = globalAveragePooling1dLayer('Name', 'gap');
gmp = globalMaxPooling1dLayer('Name', 'gmp');
poolConcat = concatenationLayer(1, 2, 'Name', 'pool_concat');

lgraph = addLayers(lgraph, finalLN);
lgraph = addLayers(lgraph, gap);
lgraph = addLayers(lgraph, gmp);
lgraph = addLayers(lgraph, poolConcat);

lgraph = connectLayers(lgraph, currentLayer, 'final_ln');
lgraph = connectLayers(lgraph, 'final_ln', 'gap');
lgraph = connectLayers(lgraph, 'final_ln', 'gmp');
lgraph = connectLayers(lgraph, 'gap', 'pool_concat/in1');
lgraph = connectLayers(lgraph, 'gmp', 'pool_concat/in2');

%% 6. 分类头
classificationHead = [
    fullyConnectedLayer(embedDim, 'Name', 'cls_proj1')
    layerNormalizationLayer('Name', 'cls_ln1')
    reluLayer('Name', 'cls_relu1')
    dropoutLayer(dropout, 'Name', 'cls_drop1')
    fullyConnectedLayer(numClasses, 'Name', 'cls_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

lgraph = addLayers(lgraph, classificationHead);
lgraph = connectLayers(lgraph, 'pool_concat', 'cls_proj1');

end

%% 辅助函数：添加Transformer块
function lgraph = addTransformerBlock(lgraph, inputLayerName, blockIdx, numHeads, embedDim, ffnDim, dropout)

blockName = sprintf('enc%d_', blockIdx);

% Pre-LN + Self-Attention
preLN1 = layerNormalizationLayer('Name', [blockName 'pre_ln1']);
lgraph = addLayers(lgraph, preLN1);
lgraph = connectLayers(lgraph, inputLayerName, [blockName 'pre_ln1']);

attnLayer = selfAttentionLayer(numHeads, embedDim, ...
    'Name', [blockName 'attn'], ...
    'DropoutProbability', dropout);
lgraph = addLayers(lgraph, attnLayer);
lgraph = connectLayers(lgraph, [blockName 'pre_ln1'], [blockName 'attn']);

% Residual 1
add1 = additionLayer(2, 'Name', [blockName 'add1']);
lgraph = addLayers(lgraph, add1);
lgraph = connectLayers(lgraph, inputLayerName, [blockName 'add1/in1']);
lgraph = connectLayers(lgraph, [blockName 'attn'], [blockName 'add1/in2']);

% Pre-LN + FFN
preLN2 = layerNormalizationLayer('Name', [blockName 'pre_ln2']);
lgraph = addLayers(lgraph, preLN2);
lgraph = connectLayers(lgraph, [blockName 'add1'], [blockName 'pre_ln2']);

ffnLayers = [
    fullyConnectedLayer(ffnDim, 'Name', [blockName 'ffn1'])
    reluLayer('Name', [blockName 'relu'])
    dropoutLayer(dropout, 'Name', [blockName 'ffn_drop'])
    fullyConnectedLayer(embedDim, 'Name', [blockName 'ffn2'])
];
lgraph = addLayers(lgraph, ffnLayers);
lgraph = connectLayers(lgraph, [blockName 'pre_ln2'], [blockName 'ffn1']);

% Residual 2
add2 = additionLayer(2, 'Name', [blockName 'add2']);
lgraph = addLayers(lgraph, add2);
lgraph = connectLayers(lgraph, [blockName 'add1'], [blockName 'add2/in1']);
lgraph = connectLayers(lgraph, [blockName 'ffn2'], [blockName 'add2/in2']);

end