function val = DSMI(state1, state2, cond)
    if nargin < 3 || isempty(cond)
        hX = h(state1);
        hY = h(state2);
        joint_data = [state1(:), state2(:)];   
        [~, ~, uidx] = unique(joint_data, 'rows');
        p_xy = accumarray(uidx, 1) / size(joint_data, 1);
        hXY = -sum(p_xy .* log2(p_xy)); 
        iXY = hX + hY - hXY;
        val = iXY;
    else
        h_f1_f2 = h_multivariate([state1, state2]);
        h_l_f2 = h_multivariate([cond, state2]);
        h_f2 = h(state2);
        h_f1_l_f2 = h_multivariate([state1, cond, state2]);
    
        iXYZ = h_f1_f2 + h_l_f2 - h_f2 - h_f1_l_f2;
        val = iXYZ;
    end
end

function H = h_multivariate(data)
    [~, ~, uidx] = unique(data, 'rows');
    p = accumarray(uidx, 1) / size(data, 1);
    H = -sum(p .* log2(p));
end
