function val = DSMI(target_state_Sic, state_Sjd, conditioning_set_S)
% DSMI: Differential State Mutual Information
    h_YZ = h_multivariate([state_Sjd, conditioning_set_S]);
    h_XZ = h_multivariate([target_state_Sic, conditioning_set_S]);
    h_Z  = h(conditioning_set_S);
    h_YXZ = h_multivariate([state_Sjd, target_state_Sic, conditioning_set_S]);
    
    val = h_YZ + h_XZ - h_Z - h_YXZ;
end
function H = h_multivariate(data)
    [~, ~, uidx] = unique(data, 'rows');
    p = accumarray(uidx, 1) / size(data, 1);
    H = -sum(p .* log2(p));
end

function H = h(data)
    if isempty(data)
        H = 0;
        return;
    end
    H = h_multivariate(data);
end
