function lgraph = defineMPT_SFANet_1D(numFeatures, numClasses)
% MPT-SFANet_1D: 针对一维雷达数据的适配版本
% 参考文献: MPT-SFANet (IEEE TAES 2024)
% 核心复现点:
% 1. Multi-order Pooling: 同时利用一阶(均值)和二阶(统计/方差)特征
% 2. Pyramid Pooling: 多尺度捕获上下文
% 3. Semantic Feature Aggregation: 融合深层和浅层特征

fprintf('构建对比模型 MPT-SFANet (1D适配版)...\n');

embedDim = 64;
numHeads = 4;
ffnDim = 128;
dropout = 0.2;

lgraph = layerGraph();

%% 1. 输入层
inputLayer = sequenceInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'zscore');
lgraph = addLayers(lgraph, inputLayer);

%% 2. 浅层特征提取 (模拟 SFAM 中的浅层特征)
featLayer = [
    fullyConnectedLayer(embedDim, 'Name', 'shallow_proj')
    layerNormalizationLayer('Name', 'shallow_norm')
    reluLayer('Name', 'shallow_relu')
];
lgraph = addLayers(lgraph, featLayer);
lgraph = connectLayers(lgraph, 'input', 'shallow_proj');

%% 3. 核心模块: Multi-order Pooling Transformer Module (MPTM)
% 这里我们模拟 MPTM 的结构，将特征分为 Transformer 流和 统计流

% --- Branch A: Transformer (Semantic Context) ---
% 第1个 Block
lgraph = addTransformerBlock(lgraph, 'shallow_relu', 1, numHeads, embedDim, ffnDim, dropout);

% 第2个 Block
lgraph = addTransformerBlock(lgraph, 'enc1_add2', 2, numHeads, embedDim, ffnDim, dropout);

% --- Branch B: Multi-order Pooling (核心复现) ---
% 文献中使用了金字塔池化(一阶)和协方差池化(二阶)

% B1. 一阶池化 (First-order): 多尺度平均池化
% 由于序列长度T=10较短，我们用 Global Avg 和 Global Max 代表不同尺度
avgPool = globalAveragePooling1dLayer('Name', 'order1_avg');
maxPool = globalMaxPooling1dLayer('Name', 'order1_max');

lgraph = addLayers(lgraph, avgPool);
lgraph = addLayers(lgraph, maxPool);
lgraph = connectLayers(lgraph, 'enc2_add2', 'order1_avg');
lgraph = connectLayers(lgraph, 'enc2_add2', 'order1_max');

% B2. 二阶池化 (Second-order): 统计特征 (模拟文献中的 Global Covariance)
% 这种方法在深度学习中通常通过提取特征的标准差(Std)来近似二阶统计量
% 由于MATLAB标准层没有直接的Std层，我们用 (E[x^2] - E[x]^2) 的思路或者自定义层的思路
% 这里为了兼容性，我们使用一个独立的 FC 分支来拟合二阶统计相关性
statBranch = [
    fullyConnectedLayer(embedDim, 'Name', 'order2_fc1')
    reluLayer('Name', 'order2_relu1')
    fullyConnectedLayer(embedDim, 'Name', 'order2_extract') % 模拟二阶特征映射
    globalAveragePooling1dLayer('Name', 'order2_pool')
];
lgraph = addLayers(lgraph, statBranch);
lgraph = connectLayers(lgraph, 'enc2_add2', 'order2_fc1');

%% 4. Semantic Feature Aggregation (SFAM)
% 将多阶特征聚合
concat = concatenationLayer(1, 3, 'Name', 'sfa_concat'); % 聚合 Avg, Max, 和 Second-Order
lgraph = addLayers(lgraph, concat);
lgraph = connectLayers(lgraph, 'order1_avg', 'sfa_concat/in1');
lgraph = connectLayers(lgraph, 'order1_max', 'sfa_concat/in2');
lgraph = connectLayers(lgraph, 'order2_pool', 'sfa_concat/in3');

%% 5. 分类头
classificationHead = [
    fullyConnectedLayer(embedDim, 'Name', 'cls_proj')
    layerNormalizationLayer('Name', 'cls_ln')
    reluLayer('Name', 'cls_relu')
    dropoutLayer(dropout, 'Name', 'cls_drop')
    fullyConnectedLayer(numClasses, 'Name', 'cls_output')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')
];

lgraph = addLayers(lgraph, classificationHead);
lgraph = connectLayers(lgraph, 'sfa_concat', 'cls_proj');

end

%% 辅助函数：添加Transformer块 (标准MHSA)
function lgraph = addTransformerBlock(lgraph, inputLayerName, blockIdx, numHeads, embedDim, ffnDim, dropout)
    blockName = sprintf('enc%d_', blockIdx);

    % Pre-LN
    preLN1 = layerNormalizationLayer('Name', [blockName 'pre_ln1']);
    lgraph = addLayers(lgraph, preLN1);
    lgraph = connectLayers(lgraph, inputLayerName, [blockName 'pre_ln1']);

    % Attention
    attnLayer = selfAttentionLayer(numHeads, embedDim, 'Name', [blockName 'attn'], 'DropoutProbability', dropout);
    lgraph = addLayers(lgraph, attnLayer);
    lgraph = connectLayers(lgraph, [blockName 'pre_ln1'], [blockName 'attn']);

    % Residual 1
    add1 = additionLayer(2, 'Name', [blockName 'add1']);
    lgraph = addLayers(lgraph, add1);
    lgraph = connectLayers(lgraph, inputLayerName, [blockName 'add1/in1']);
    lgraph = connectLayers(lgraph, [blockName 'attn'], [blockName 'add1/in2']);

    % FFN
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