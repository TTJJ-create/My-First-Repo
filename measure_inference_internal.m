function t_mean = measure_inference_internal(model, sample, type, num_trials, warmup_runs)
    % 内部测量函数，处理不同的预测方法 (predict vs classify)
    
    % 1. 预热 (Warmup)
    for i = 1:warmup_runs
        if strcmp(type, 'predict')
            predict(model, sample);
        else
            classify(model, sample);
        end
    end
    
    % 2. 正式测量 (Measurement)
    times = zeros(num_trials, 1);
    for i = 1:num_trials
        tic;
        if strcmp(type, 'predict')
            predict(model, sample);
        else
            classify(model, sample);
        end
        times(i) = toc * 1000; % 秒 -> 毫秒
    end
    
    t_mean = mean(times);
end