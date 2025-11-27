function t_mean = measure_inference_internal(model, sample, type, num_trials, warmup_runs)
    % 内部测量函数，测量推理时间
    % 处理 dlnetwork 和传统网络的输入格式差异
    
    % 判断是否是 dlnetwork，若是则转换输入为 dlarray
    if isa(model, 'dlnetwork')
        % dlnetwork 需要 dlarray 输入
        if iscell(sample)
            % cell 格式（序列数据）：转换为 dlarray
            sample_input = dlarray(sample{1}, 'TCB'); % Time, Channel, Batch
        else
            % 普通数组：转换为 dlarray
            if isrow(sample)
                sample_input = dlarray(sample', 'CB'); % Channel, Batch
            else
                sample_input = dlarray(sample, 'CB');
            end
        end
    else
        % 传统网络：保持原格式
        sample_input = sample;
    end
    
    % 1. 预热 (Warmup)
    for i = 1:warmup_runs
        predict(model, sample_input);
    end
    
    % 2. 正式测量 (Measurement)
    times = zeros(num_trials, 1);
    for i = 1:num_trials
        tic;
        predict(model, sample_input);
        times(i) = toc * 1000; % 秒 -> 毫秒
    end
    
    t_mean = mean(times);
end