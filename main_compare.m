close all;
clear;
clc;

%% ========================================================================
%% 1. KHỞI TẠO HỆ THỐNG
%% ========================================================================
fprintf('Initializing System...\n');
M = 12;             
Q = 160;            
phi = 1;
eqDir = -1:phi/Q:1-phi/Q; 
desDirs_c = 0.0;    

% Giả định các hàm này đã có trong thư mục làm việc
Aq = generateQuantizedArrResponse(M, eqDir);
[PdM, P_refGen, W0_init] = generateDesPattern(eqDir, sin(desDirs_c), Aq);

% Chỉ số alpha
alpha = sort([find(ismember(eqDir, eqDir(1:4:end))), find(PdM)]);

% Chuẩn hóa mẫu mục tiêu
target_pattern = PdM(alpha) ./ max(PdM(alpha));

% --- TRỌNG SỐ OPTIMIZATION ---
weights = ones(size(alpha));
sidelobe_indices_opt = abs(eqDir(alpha)) > 0.22; 
weights(sidelobe_indices_opt) = 100; 

%% ========================================================================
%% 2. L-SHADE (MODIFIED FOR STRICT COMPLIANCE)
%% ========================================================================
fprintf('Running L-SHADE with Weighted Cost Function...\n');
tic;

% --- Cấu hình tham số ---
max_iter = 600;         
pop_size_init = 100;    
pop_size_min = 10;
n_vars = 2 * M; 

% [MODIFICATION 1] Calculate MAX_NFE Budget based on Linear Reduction Profile
% L-SHADE requires LPSR to be a function of NFE, not Iterations.
% Total NFE approx area under trapezoid: Iterations * (P_start + P_end) / 2
MAX_NFE = floor(max_iter * (pop_size_init + pop_size_min) / 2);
nfe_count = 0; % Counter for Function Evaluations

lb = [-pi * ones(1, M), zeros(1, M)];
ub = [ pi * ones(1, M), 2 * ones(1, M)]; 

H = 6; 
mem_CR = 0.5 * ones(H, 1); 
mem_F = 0.5 * ones(H, 1); 
mem_k = 1;

archive = []; 
arc_size_max = pop_size_init; 

pop = lb + (ub - lb) .* rand(pop_size_init, n_vars);
pop_size = pop_size_init;

calc_cost = @(x) get_weighted_error(x, Aq, alpha, target_pattern, weights, M);

cost = zeros(pop_size, 1);
for i = 1:pop_size
    cost(i) = calc_cost(pop(i, :));
end
nfe_count = nfe_count + pop_size; % Initial NFE count

[global_min, best_idx] = min(cost);
best_sol = pop(best_idx, :);

% Optimization Loop
for iter = 1:max_iter
    % [MODIFICATION 2] LPSR based on NFE (Eq. 6 in Tanabe & Fukunaga 2014)
    % Reduces population relative to the expended computational budget
    plan_pop_size = round(((pop_size_min - pop_size_init) / MAX_NFE) * nfe_count + pop_size_init);

    new_pop = pop;
    success_F = []; 
    success_CR = []; 
    diff_fit = [];
    
    [~, sorted_idx] = sort(cost);
    p_num = max(round(0.11 * pop_size), 2);
    
    % Population Loop
    for i = 1:pop_size
        idx_m = randi(H);
        mu_cr = mem_CR(idx_m); 
        mu_f = mem_F(idx_m);
        
        % Parameter Sampling
        CR = max(0, min(1, normrnd(mu_cr, 0.1)));
        while true
            F = mu_f + 0.1 * tan(pi * (rand - 0.5)); 
            if F > 0, break; end; 
        end; 
        F = min(1, F);
        
        % Mutation: current-to-pbest/1
        pbest = sorted_idx(randi(p_num));
        r1 = randi(pop_size); while r1==i, r1=randi(pop_size); end
        comm_pop = [pop; archive];
        r2 = randi(size(comm_pop, 1)); while r2==i || r2==r1, r2=randi(size(comm_pop, 1)); end
        
        mutant = pop(i,:) + F*(pop(pbest,:) - pop(i,:)) + F*(pop(r1,:) - comm_pop(r2,:));
        
        % [MODIFICATION 3] Boundary Handling: Midpoint-Target Rule
        % 1. Phases: Cyclic topology (Wrapping is mathematically valid)
        mutant(1:M) = angle(exp(1j * mutant(1:M))); 
        
        % 2. Amplitudes: Midpoint Rule (Ref: Tanabe & Fukunaga 2013, Eq. 6 constraint handling)
        idx_amp = M+1:n_vars;
        mut_amp = mutant(idx_amp);
        parent_amp = pop(i, idx_amp);
        
        % Check Lower Bound (0)
        viol_lb = mut_amp < 0;
        if any(viol_lb)
            mut_amp(viol_lb) = (0 + parent_amp(viol_lb)) / 2;
        end
        
        % Check Upper Bound (2)
        viol_ub = mut_amp > 2;
        if any(viol_ub)
            mut_amp(viol_ub) = (2 + parent_amp(viol_ub)) / 2;
        end
        mutant(idx_amp) = mut_amp;
        
        % Crossover
        jrand = randi(n_vars);
        mask = rand(1, n_vars) < CR; 
        mask(jrand) = 1;
        trial = pop(i,:); 
        trial(mask) = mutant(mask);
        
        % Selection
        f_trial = calc_cost(trial);
        nfe_count = nfe_count + 1; % Increment NFE per evaluation
        
        if f_trial <= cost(i)
            if f_trial < cost(i)
                archive = [archive; pop(i,:)];
                success_F = [success_F; F]; 
                success_CR = [success_CR; CR];
                diff_fit = [diff_fit; cost(i) - f_trial];
            end
            new_pop(i,:) = trial; 
            cost(i) = f_trial;
            if f_trial < global_min, global_min = f_trial; best_sol = trial; end
        end
    end
    pop = new_pop;
    
    % Update Archive
    if size(archive, 1) > arc_size_max
        archive = archive(randperm(size(archive, 1), arc_size_max), :); 
    end
    
    % Update Memory
    if ~isempty(success_F)
        w = diff_fit / sum(diff_fit);
        mem_CR(mem_k) = sum(w .* success_CR);
        mem_F(mem_k) = (w' * (success_F.^2)) / (w' * success_F);
        mem_k = mod(mem_k, H) + 1;
    end
    
    % Linear Population Size Reduction (Applied at end of generation)
    if pop_size > plan_pop_size
        [~, worst] = sort(cost, 'descend');
        num_rem = pop_size - plan_pop_size;
        pop(worst(1:num_rem), :) = []; 
        cost(worst(1:num_rem)) = [];
        pop_size = plan_pop_size;
        arc_size_max = pop_size; % Resize archive limit as per L-SHADE definition
    end
    
    if mod(iter, 100) == 0
        fprintf('Iter %d | NFE %d | Min Weighted Error: %.6f\n', iter, nfe_count, global_min);
    end
    
    % Safety break if NFE exceeds budget (though LPSR prevents divergence)
    if nfe_count >= MAX_NFE
        break;
    end
end

best_phases = best_sol(1:M);
best_amps = best_sol(M+1:end);
W_LSHADE = best_amps' .* exp(1j * best_phases');
t_LSHADE = toc;

%% ========================================================================
%% 3. TÍNH TOÁN VÀ VẼ
%% ========================================================================
% --- ULA ---
W_ULA = ones(M, 1); 
P_ULA_dB = 20*log10(abs(W_ULA' * Aq) / max(abs(W_ULA' * Aq)));

% --- ILS ---
W_ILS = twoStepILS(50, alpha, Aq, W0_init, ones(size(eqDir)), PdM);
P_ILS_dB = 20*log10(abs(W_ILS' * Aq) / max(abs(W_ILS' * Aq)));

% --- L-SHADE ---
P_LSHADE_dB = 20*log10(abs(W_LSHADE' * Aq) / max(abs(W_LSHADE' * Aq)));

% --- TÍNH TOÁN HPBW (Half-Power Beamwidth) ---
hpbw_ula = get_HPBW(eqDir, P_ULA_dB);
hpbw_ils = get_HPBW(eqDir, P_ILS_dB);
hpbw_lshade = get_HPBW(eqDir, P_LSHADE_dB);

% --- TÍNH SLL (Mask hiển thị) ---
mask_sll_display = abs(eqDir) > 0.3; 
sll_ula = max(P_ULA_dB(mask_sll_display));
sll_ils = max(P_ILS_dB(mask_sll_display));
sll_lshade = max(P_LSHADE_dB(mask_sll_display));

% --- BẢNG SO SÁNH KẾT QUẢ ---
fprintf('\n======================================================\n');
fprintf('                COMPARISON TABLE                      \n');
fprintf('======================================================\n');
Method = {'ULA'; 'ILS'; 'L-SHADE'};
SLL_dB = [sll_ula; sll_ils; sll_lshade];
HPBW_u = [hpbw_ula; hpbw_ils; hpbw_lshade];
% Tạo bảng hiển thị
ResultsTable = table(SLL_dB, HPBW_u, 'RowNames', Method);
disp(ResultsTable);
fprintf('======================================================\n');

% --- VẼ ĐỒ THỊ ---
figure('Color', 'w', 'Position', [100 100 1000 600]);
hold on; grid on; box on;

% Desired (Hồng)
P_Desired_dB = 20*log10(PdM/max(PdM));
P_Desired_dB(P_Desired_dB < -100) = -100;
plot(eqDir, P_Desired_dB, 'm:', 'LineWidth', 1.5, 'DisplayName', 'Desired');

% ULA (Xám)
plot(eqDir, P_ULA_dB, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', sprintf('ULA (HPBW: %.4f)', hpbw_ula));

% ILS (Đen đứt nét)
plot(eqDir, P_ILS_dB, '--k', 'LineWidth', 2, 'DisplayName', sprintf('ILS (HPBW: %.4f)', hpbw_ils));

% L-SHADE (Đỏ liền)
plot(eqDir, P_LSHADE_dB, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('L-SHADE (HPBW: %.4f)', hpbw_lshade));

xlabel('Equivalent Directions (u = sin\theta)');
ylabel('Normalized Magnitude (dB)');
title(sprintf('Beamforming Synthesis Comparison (M=%d)', M));
legend('Location', 'northeast');
xlim([-1 1]); ylim([-45 2]);

% Hiển thị kết quả chi tiết trên đồ thị
txt = { '      --- PERFORMANCE ---', ...
        sprintf('ULA: SLL=%.2f dB, HPBW=%.4f', sll_ula, hpbw_ula), ...
        sprintf('ILS: SLL=%.2f dB, HPBW=%.4f', sll_ils, hpbw_ils), ...
        sprintf('L-SHADE: SLL=%.2f dB, HPBW=%.4f', sll_lshade, hpbw_lshade) };
text(-0.95, -5, txt, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'FontName', 'Consolas', 'FontSize', 9);

%% ========================================================================
%% CÁC HÀM PHỤ TRỢ (HELPER FUNCTIONS)
%% ========================================================================

% 1. Hàm tính lỗi Optimization
function err = get_weighted_error(x, Aq, alpha, target, weights, M)
    phases = x(1:M);
    amps = x(M+1:end);
    w = amps' .* exp(1j * phases');
    
    generated = abs(w' * Aq(:, alpha));
    if max(generated) == 0
        generated_norm = generated; 
    else
        generated_norm = generated / max(generated);
    end
    
    diff = abs(generated_norm - target);
    weighted_diff = diff .* weights;
    err = norm(weighted_diff);
end

% 2. Hàm tính HPBW (Half-Power Beamwidth)
function bw = get_HPBW(u_axis, pattern_dB)
    [~, max_idx] = max(pattern_dB);
    
    % Tìm bên TRÁI
    left_region_vals = pattern_dB(1:max_idx);
    left_region_u = u_axis(1:max_idx);
    idx_L = find(left_region_vals <= -3, 1, 'last');
    
    if isempty(idx_L) || idx_L == length(left_region_u)
        u_left = u_axis(1); 
    else
        y1 = left_region_vals(idx_L);
        y2 = left_region_vals(idx_L+1);
        x1 = left_region_u(idx_L);
        x2 = left_region_u(idx_L+1);
        u_left = x1 + (-3 - y1) * (x2 - x1) / (y2 - y1);
    end
    
    % Tìm bên PHẢI
    right_region_vals = pattern_dB(max_idx:end);
    right_region_u = u_axis(max_idx:end);
    idx_R = find(right_region_vals <= -3, 1, 'first');
    
    if isempty(idx_R)
        u_right = u_axis(end); 
    else
        y1 = right_region_vals(idx_R-1);
        y2 = right_region_vals(idx_R);
        x1 = right_region_u(idx_R-1);
        x2 = right_region_u(idx_R);
        u_right = x1 + (-3 - y1) * (x2 - x1) / (y2 - y1);
    end
    
    bw = u_right - u_left;
end