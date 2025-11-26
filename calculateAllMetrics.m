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
