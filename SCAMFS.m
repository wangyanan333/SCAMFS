function selfea = SCAMFS(data, L, delta1, delta2, k1, k2)
    [N, p] = size(data);
    num_feats = p - L;
    
    label_indices = (num_feats + 1) : p;
    feature_indices = 1 : num_feats;

    feat_data = data(:, 1:num_feats);
    label_classes = cell(1, L);
    state_indices_by_label = cell(1, L);
    flattened_labels = struct('orig_lbl_idx', {}, 'class_val', {}, 'bin_vec', {});
    class_count = 0;
    for i_cache = 1:L
        lbl_idx = label_indices(i_cache);
        u_vals = unique(data(:, lbl_idx));
        label_classes{i_cache} = u_vals;
        state_indices_by_label{i_cache} = zeros(1, length(u_vals));
        for d_idx = 1:length(u_vals)
            class_count = class_count + 1;
            state_indices_by_label{i_cache}(d_idx) = class_count;
            flattened_labels(class_count).orig_lbl_idx = lbl_idx;
            flattened_labels(class_count).class_val = u_vals(d_idx);
            flattened_labels(class_count).bin_vec = double(data(:, lbl_idx) == u_vals(d_idx));
        end
    end
    num_total_classes = class_count;

    MI_state_feature = zeros(num_total_classes, num_feats);
    for s_idx = 1:num_total_classes
        state_vec = flattened_labels(s_idx).bin_vec;
        for f_idx = feature_indices
            MI_state_feature(s_idx, f_idx) = SSMI(state_vec, feat_data(:, f_idx));
        end
    end
    
    global_selected_features = [];

    for i = 1:L
        current_lbl_idx = label_indices(i);
        unique_classes = label_classes{i};
        
        for c_idx = 1:length(unique_classes)
            target_val = unique_classes(c_idx);
            
            target_vec = double(data(:, current_lbl_idx) == target_val);
            
            if sum(target_vec) < 5, continue; end

            target_state_idx = state_indices_by_label{i}(c_idx);
            mi_target_feats = MI_state_feature(target_state_idx, :);

            Matrix_Ric = []; 
            
            cand_label_classes = []; 
            
            for cache_idx = 1:num_total_classes
                if flattened_labels(cache_idx).orig_lbl_idx == current_lbl_idx
                    continue;
                end
                val = DSMI(target_vec, flattened_labels(cache_idx).bin_vec);
                if val > delta2
                    cand_label_classes = [cand_label_classes; ...
                        flattened_labels(cache_idx).orig_lbl_idx, ...
                        flattened_labels(cache_idx).class_val, val, cache_idx];
                end
            end

            if ~isempty(cand_label_classes)
                cand_label_classes = sortrows(cand_label_classes, -3);
                
                temp_cand_vecs = zeros(N, size(cand_label_classes, 1));
                for k = 1:size(cand_label_classes, 1)
                     temp_cand_vecs(:, k) = flattened_labels(cand_label_classes(k, 4)).bin_vec;
                end
                
                selected_indices = [];
                for k = 1:size(temp_cand_vecs, 2)
                    current_cand_vec = temp_cand_vecs(:, k);
                    
                    if isempty(selected_indices)
                        selected_indices = [selected_indices, k];
                    else
                        existing_data = temp_cand_vecs(:, selected_indices);
                        val = DSMI(target_vec, current_cand_vec, existing_data);
                        
                        if val > delta2
                            selected_indices = [selected_indices, k];
                        end
                    end
                end
                
                Matrix_Ric = temp_cand_vecs(:, selected_indices);
            end

            Cand_PC = [];
            for f_idx = feature_indices
                val = mi_target_feats(f_idx);
                if val > delta1
                    Cand_PC = [Cand_PC; f_idx, val];
                end
            end
            
            PC_indices = [];
            if ~isempty(Cand_PC)
                Cand_PC = sortrows(Cand_PC, -2);
                num_keep_k1 = ceil(size(Cand_PC, 1) * k1);
                Cand_PC = Cand_PC(1:num_keep_k1, :);
                current_PC = Cand_PC(:, 1)';
                
                final_PC = current_PC;
                for f_k = current_PC
                    S = setdiff(final_PC, f_k);
                    if isempty(S)
                        Data_Cond = Matrix_Ric;
                    else
                        Data_Cond = [feat_data(:, S), Matrix_Ric];
                    end
                    
                    if isempty(Data_Cond)
                        val = mi_target_feats(f_k);
                    else
                        val = SSMI(target_vec, feat_data(:, f_k), Data_Cond);
                    end
                    
                    if val <= delta1
                        final_PC(final_PC == f_k) = [];
                    end
                end
                PC_indices = final_PC;
            end
            
            candidates_Z = setdiff(feature_indices, PC_indices);
            v1_cache = zeros(1, num_feats);
            if isempty(Matrix_Ric)
                v1_cache(candidates_Z) = mi_target_feats(candidates_Z);
            else
                for z_node = candidates_Z
                    v1_cache(z_node) = SSMI(target_vec, feat_data(:, z_node), Matrix_Ric);
                end
            end
            spouse_candidates = candidates_Z(v1_cache(candidates_Z) <= delta1);
            SP_mask = false(1, num_feats);
            
            for x_node = PC_indices
                if isempty(Matrix_Ric)
                    Cond_XR = feat_data(:, x_node);
                else
                    Cond_XR = [feat_data(:, x_node), Matrix_Ric];
                end

                for z_node = spouse_candidates
                    val2 = SSMI(target_vec, feat_data(:, z_node), Cond_XR);
                    
                    if val2 > delta1
                        SP_mask(z_node) = true;
                    end
                end
            end
            SP_indices = find(SP_mask);

            CMB_indices = unique([PC_indices, SP_indices]);
            
            if ~isempty(Matrix_Ric)
                num_ric = size(Matrix_Ric, 2);
                keep_ric_mask = true(1, num_ric);
                
                F_miss_candidates = setdiff(feature_indices, CMB_indices);
                S_base_cache = cell(1, num_ric);
                val_block_cache = zeros(1, num_ric);
                for r_i = 1:num_ric
                    other_ric_idx = setdiff(1:num_ric, r_i);
                    S_base_cache{r_i} = [feat_data(:, PC_indices), Matrix_Ric(:, other_ric_idx)];
                    Y_block_vec = Matrix_Ric(:, r_i);
                    if isempty(S_base_cache{r_i})
                        val_block_cache(r_i) = DSMI(target_vec, Y_block_vec);
                    else
                        val_block_cache(r_i) = DSMI(target_vec, Y_block_vec, S_base_cache{r_i});
                    end
                end
                
                for f_miss = F_miss_candidates
                    if mi_target_feats(f_miss) < delta1, continue; end
                    
                    replaced_flag = false;
                    for r_i = 1:num_ric
                        if ~keep_ric_mask(r_i), continue; end
                        
                        Data_S_base = S_base_cache{r_i};

                        if isempty(Data_S_base)
                            val_feat = mi_target_feats(f_miss);
                        else
                            val_feat = SSMI(target_vec, feat_data(:, f_miss), Data_S_base);
                        end
                        val_block = val_block_cache(r_i);
                        
                        if val_feat > val_block
                            CMB_indices = [CMB_indices, f_miss];
                            keep_ric_mask(r_i) = false;
                            replaced_flag = true;
                            break; 
                        end
                    end
                    if replaced_flag, break; end
                end
            end
            
            current_CMB = CMB_indices;
            scores = zeros(length(current_CMB), 1);
            for k = 1:length(current_CMB)
                elem = current_CMB(k);
                scores(k) = mi_target_feats(elem);
            end
            [~, sort_idx] = sort(scores, 'descend');
            sorted_CMB = current_CMB(sort_idx);
            
            num_keep = ceil(length(sorted_CMB) * k2);
            top_CMB = sorted_CMB(1:num_keep);
            
            final_CMB_symmetry = top_CMB(mi_target_feats(top_CMB) > delta1);
            CMB_indices = final_CMB_symmetry;

            final_MB_clean = CMB_indices;
            for f_k = CMB_indices
                val_current = mi_target_feats(f_k);
                is_redundant = false;
                
                for cache_idx = 1:num_total_classes
                    if flattened_labels(cache_idx).orig_lbl_idx == current_lbl_idx
                        continue;
                    end
                    val_other = MI_state_feature(cache_idx, f_k); 
                    
                    if val_other > (val_current * 1.2)
                        is_redundant = true;
                        break;
                    end
                end
                
                if is_redundant
                    final_MB_clean(final_MB_clean == f_k) = [];
                end
            end

            global_selected_features = [global_selected_features, final_MB_clean];
            
        end 
    end
    
    selfea = unique(global_selected_features);
end
