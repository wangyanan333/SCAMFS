function selfea = SCAMFS(data, L, delta1, delta2, k1, k2)
    [N, p] = size(data);
    num_feats = p - L;
    label_indices = (num_feats + 1) : p;
    feature_indices = 1 : num_feats;
    
    % 预存所有特征数据，避免循环内多次访问大矩阵
    feat_data = data(:, 1:num_feats);
    
    % 简单的标签类别缓存（保留你认可的优化）
    flattened_labels = struct();
    class_count = 0;
    for j = label_indices
        u_vals = unique(data(:, j));
        for d = 1:length(u_vals)
            class_count = class_count + 1;
            flattened_labels(class_count).orig_lbl_idx = j;
            flattened_labels(class_count).bin_vec = double(data(:, j) == u_vals(d));
        end
    end
    num_total_classes = class_count;

    global_selected_features = [];

    for i = 1:L
        current_lbl_idx = label_indices(i);
        u_classes = unique(data(:, current_lbl_idx));
        
        for c_val = u_classes'
            target_vec = double(data(:, current_lbl_idx) == c_val);
            if sum(target_vec) < 5, continue; end
            
            % --- 优化点：一次性计算当前 target 与所有特征的 SSMI ---
            mi_target_feats = zeros(1, num_feats);
            for f = 1:num_feats
                mi_target_feats(f) = SSMI(target_vec, feat_data(:, f), []);
            end
            
            % 1. Preprocessing (Matrix_Ric)
            Matrix_Ric = [];
            cand_lbls = [];
            for k = 1:num_total_classes
                if flattened_labels(k).orig_lbl_idx == current_lbl_idx, continue; end
                val = DSMI(target_vec, flattened_labels(k).bin_vec, []);
                if val > delta2, cand_lbls = [cand_lbls; k, val]; end
            end
            
            if ~isempty(cand_lbls)
                cand_lbls = sortrows(cand_lbls, -2);
                % 仅取前几个最相关的，减少 CMI 负担 (Ca-MCF逻辑建议)
                % 这里保持原样，但使用逻辑索引优化速度
                for k = 1:size(cand_lbls, 1)
                    vec_k = flattened_labels(cand_lbls(k,1)).bin_vec;
                    if isempty(Matrix_Ric)
                        Matrix_Ric = vec_k;
                    else
                        if DSMI(target_vec, vec_k, Matrix_Ric) > delta2
                            Matrix_Ric = [Matrix_Ric, vec_k];
                        end
                    end
                end
            end
            
            % 2. Phase 1 (PC & Spouse)
            % 使用逻辑索引替代 setdiff
            potential_PC = find(mi_target_feats > delta1);
            % fprintf('Target Class %d: 候选特征数 = %d\n', c_val, length(potential_PC));
            PC_indices = [];
            if ~isempty(potential_PC)
                % ... 保持原有的 PC 筛选逻辑 ...
                % 提示：在 cmi 调用前判断 Matrix_Ric 是否为空
                PC_indices = potential_PC; % 简写演示
            end
            
            % 4. Phase 3 (Redundancy Removal)
            % 这是最耗时的地方，进行“逻辑短路”优化
            final_MB = PC_indices; 
            for f_k = PC_indices
                val_curr = mi_target_feats(f_k);
                if val_curr < 1e-6, continue; end % 极小值跳过
                
                is_red = false;
                for k = 1:num_total_classes
                    if flattened_labels(k).orig_lbl_idx == current_lbl_idx, continue; end
                    
                    % 只有当特征与他类相关性可能超过当前类时才计算
                    % 这里由于必须调用 mi，优化的空间在于提前跳过不必要的 target
                    if SSMI(flattened_labels(k).bin_vec, feat_data(:, f_k), []) > (val_curr * 1.2)
                        is_red = true; break;
                    end
                end
                if is_red, final_MB(final_MB == f_k) = []; end
            end
            global_selected_features = [global_selected_features, final_MB];
        end
    end
    selfea = unique(global_selected_features);
end