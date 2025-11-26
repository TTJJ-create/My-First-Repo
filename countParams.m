%% 辅助函数：智能计算网络参数量 (兼容 DAGNetwork 和 dlnetwork)
function numParams = countParams(net)
    numParams = 0;
    
    try
        % 情况 1: 已经是 dlnetwork (有 Learnables 属性)
        if isa(net, 'dlnetwork')
            p = net.Learnables.Value;
            numParams = sum(cellfun(@numel, p));
            
        % 情况 2: 是 DAGNetwork 或 SeriesNetwork (无 Learnables 属性)
        elseif isa(net, 'DAGNetwork') || isa(net, 'SeriesNetwork')
            % 方法 A: 尝试转为 dlnetwork (最准，支持 R2020b+)
            try
                dlnet = dlnetwork(net);
                p = dlnet.Learnables.Value;
                numParams = sum(cellfun(@numel, p));
            catch
                % 方法 B: 如果转换失败 (旧版本)，则遍历层累加
                layers = net.Layers;
                for i = 1:numel(layers)
                    ly = layers(i);
                    % 累加常见可学习参数 (Weights, Bias 等)
                    props = properties(ly);
                    for j = 1:numel(props)
                        propName = props{j};
                        % 检查属性名是否包含权重或偏置关键字
                        if contains(propName, {'Weights', 'Bias', 'Scale', 'Offset', 'RecurrentWeights', 'InputWeights'})
                            val = ly.(propName);
                            % 确保是数值参数且非空
                            if isnumeric(val) && ~isempty(val)
                                numParams = numParams + numel(val);
                            end
                        end
                    end
                end
            end
            
        % 情况 3: 传统机器学习模型 (如 SVM)
        elseif isprop(net, 'SupportVectors')
            % 粗略估算: 支持向量数 * 特征维数
            numParams = sum(cellfun(@(x) size(x,1), net.SupportVectors)) * size(net.SupportVectors{1}, 2);
            
        else
            fprintf('  (未知网络类型，跳过参数统计)\n');
            return;
        end
        
        % 输出结果
        fprintf('模型参数量: %.2f M (%.0f)\n', numParams/1e6, numParams);
        
    catch ME
        fprintf('  (参数计算出错: %s)\n', ME.message);
        numParams = 0;
    end
end