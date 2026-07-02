function val = SSMI(state, feature, cond)
    if nargin < 3 || isempty(cond)
        hX = h(state);
        hY = h(feature);
        joint_data = [state(:), feature(:)];   
        [~, ~, uidx] = unique(joint_data, 'rows');
        p_xy = accumarray(uidx, 1) / size(joint_data, 1);
        hXY = -sum(p_xy .* log2(p_xy)); 
        iXY = hX + hY - hXY;
        val = iXY;
    else
        h_f1_f2 = h_multivariate([state, feature]);
        h_l_f2 = h_multivariate([cond, feature]);
        h_f2 = h(feature);
        h_f1_l_f2 = h_multivariate([state, cond, feature]);
        iXYZ = h_f1_f2 + h_l_f2 - h_f2 - h_f1_l_f2;
        val = iXYZ;
    end
end

function H = h_multivariate(data)
    [~, ~, uidx] = unique(data, 'rows');
    p = accumarray(uidx, 1) / size(data, 1);
    H = -sum(p .* log2(p));
end
