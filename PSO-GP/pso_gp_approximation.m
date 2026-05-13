% =========================================================================
% PSO-GP: Evolutionary Optimization-based Activation Function Approximation
% =========================================================================
% Description:
%   This script uses Particle Swarm Optimization (PSO) to find optimal
%   interval partitions and Genetic Programming (GP) to approximate any
%   nonlinear activation function using piecewise linear segments.
%
% Usage:
%   1. Set the activation function in the 'User Configuration' section
%   2. Set the number of intervals/segments
%   3. Run the script
%
% Output:
%   - Optimal interval boundaries
%   - Corresponding linear equations: L_j(x) = a_j*x + b_j for each segment
%   - MAE for each segment and overall
%
% Requirements:
%   - GP-OLS Toolbox: http://www.fmt.veim.hu/softcomp
%     (gpols_init, gpols_evaluate, gpols_mainloop, gpols_result)
%
% Authors: Mahendra Kumar Gurve, Gaurav Kumar et al.
%          Indian Institute of Technology Jammu, India
% =========================================================================

clc; clear; close all;

%% -------------------------------------------------------------------------
%  USER CONFIGURATION — Edit this section only
% -------------------------------------------------------------------------

% --- Activation Function ---
% Choose one: 'sigmoid', 'tanh', 'gelu', or define your own below
ACTIVATION_FUNCTION = 'sigmoid';

% --- Number of Intervals/Segments ---
NUM_INTERVALS = 13;

% --- Input Domain ---
LOWER_BOUND = 0;   % Start of approximation domain
UPPER_BOUND = 8;   % End of approximation domain

% --- PSO Parameters ---
NUM_PARTICLES  = 30;
NUM_ITERATIONS = 20;
INERTIA        = 0.5;
COGNITIVE      = 1.5;
SOCIAL         = 1.5;

% --- GP Parameters ---
GP_POP_SIZE    = 50;
GP_MAX_DEPTH   = 2;
GP_GENERATIONS = 10;
GP_OPTIONS     = [0.8 0.7 0.3 2 10 0.2 50 0.05 0 0];

% --- Output ---
SAVE_RESULTS = true;
OUTPUT_FILE  = 'pso_gp_results.txt';

%% -------------------------------------------------------------------------
%  ACTIVATION FUNCTION DEFINITION
% -------------------------------------------------------------------------

switch lower(ACTIVATION_FUNCTION)
    case 'sigmoid'
        act_fn  = @(x) 1 ./ (1 + exp(-x));
        fn_name = 'Sigmoid';
    case 'tanh'
        act_fn  = @(x) tanh(x);
        fn_name = 'Tanh';
    case 'gelu'
        act_fn  = @(x) 0.5 .* x .* (1 + erf(x / sqrt(2)));
        fn_name = 'GELU';
    otherwise
        error('Unknown activation function. Choose: sigmoid, tanh, gelu');
end

fprintf('=============================================================\n');
fprintf(' PSO-GP Activation Function Approximation\n');
fprintf('=============================================================\n');
fprintf(' Function  : %s\n', fn_name);
fprintf(' Intervals : %d\n', NUM_INTERVALS);
fprintf(' Domain    : [%.1f, %.1f]\n', LOWER_BOUND, UPPER_BOUND);
fprintf(' Particles : %d | Iterations: %d\n', NUM_PARTICLES, NUM_ITERATIONS);
fprintf('=============================================================\n\n');

%% -------------------------------------------------------------------------
%  MAIN: RUN PSO-GP OPTIMIZATION
% -------------------------------------------------------------------------

[best_intervals, best_mae] = run_pso_gp( ...
    NUM_PARTICLES, NUM_ITERATIONS, NUM_INTERVALS, ...
    LOWER_BOUND, UPPER_BOUND, ...
    INERTIA, COGNITIVE, SOCIAL, ...
    GP_POP_SIZE, GP_MAX_DEPTH, GP_GENERATIONS, GP_OPTIONS, ...
    act_fn);

%% -------------------------------------------------------------------------
%  EXTRACT FINAL LINEAR EQUATIONS FOR BEST INTERVALS
% -------------------------------------------------------------------------

fprintf('\n Extracting final piecewise linear equations...\n');

[equations, seg_mae] = extract_linear_equations( ...
    best_intervals, GP_POP_SIZE, GP_MAX_DEPTH, GP_GENERATIONS, GP_OPTIONS, act_fn);

%% -------------------------------------------------------------------------
%  DISPLAY RESULTS
% -------------------------------------------------------------------------

fprintf('\n=============================================================\n');
fprintf(' RESULTS: Piecewise Linear Approximation of %s\n', fn_name);
fprintf('=============================================================\n\n');
fprintf(' %-5s  %-22s  %-14s  %-14s  %-12s\n', ...
        'Seg', 'Interval', 'Slope (a)', 'Intercept (b)', 'Seg MAE');
fprintf(' %s\n', repmat('-', 1, 72));

for j = 1:length(equations)
    a   = equations(j).slope;
    b   = equations(j).intercept;
    lo  = equations(j).x_start;
    hi  = equations(j).x_end;
    mae = seg_mae(j);
    fprintf(' %-5d  [%8.4f, %8.4f]  %+14.8f  %+14.8f  %.6e\n', ...
            j, lo, hi, a, b, mae);
end

fprintf('\n Overall MAE : %.6e\n', best_mae);
fprintf(' Overall EMAX: %.6e\n', ...
        compute_emax(best_intervals, equations, act_fn, LOWER_BOUND, UPPER_BOUND));
fprintf('=============================================================\n');

%% -------------------------------------------------------------------------
%  SAVE RESULTS
% -------------------------------------------------------------------------

if SAVE_RESULTS
    save_results(OUTPUT_FILE, fn_name, NUM_INTERVALS, ...
                 LOWER_BOUND, UPPER_BOUND, best_intervals, equations, seg_mae, best_mae);
    fprintf('\n Results saved to: %s\n', OUTPUT_FILE);
end

%% =========================================================================
%  FUNCTIONS
%% =========================================================================

% -------------------------------------------------------------------------
% run_pso_gp: Main PSO-GP optimization loop
% -------------------------------------------------------------------------
function [best_intervals, best_mae] = run_pso_gp( ...
        num_particles, num_iterations, num_intervals, ...
        lb, ub, w, c1, c2, ...
        gp_pop, gp_depth, gp_gen, gp_opt, act_fn)

    % Initialize swarm
    particles  = init_particles(num_particles, num_intervals, lb, ub);
    velocities = rand(num_particles, num_intervals) * 2 - 1;

    % Evaluate initial fitness
    pbest_pos    = particles;
    pbest_scores = zeros(1, num_particles);
    for i = 1:num_particles
        pbest_scores(i) = evaluate_fitness(particles(i,:), gp_pop, gp_depth, ...
                                           gp_gen, gp_opt, act_fn);
    end

    % Global best
    [gbest_score, idx] = min(pbest_scores);
    gbest_pos = pbest_pos(idx, :);

    fprintf(' Starting PSO optimization...\n\n');

    % PSO main loop
    for iter = 1:num_iterations
        for i = 1:num_particles

            % Velocity update (Eq. 1)
            inertia   = w  * velocities(i,:);
            cognitive = c1 * rand() * (pbest_pos(i,:) - particles(i,:));
            social    = c2 * rand() * (gbest_pos      - particles(i,:));
            velocities(i,:) = inertia + cognitive + social;

            % Position update (Eq. 2)
            particles(i,:) = particles(i,:) + velocities(i,:);

            % Constraint handling: 4 steps (Algorithm 1, Lines 9-16)
            particles(i,:) = max(lb, min(particles(i,:), ub));        % Step 1: clip
            particles(i,:) = sort(particles(i,:));                     % Step 2: sort
            particles(i,:) = enforce_min_gap(particles(i,:), lb, ub); % Step 3: min gap
            particles(i,:) = max(lb, min(particles(i,:), ub));        % Step 4: re-clip

            % Fitness evaluation (Eq. 3)
            score = evaluate_fitness(particles(i,:), gp_pop, gp_depth, ...
                                     gp_gen, gp_opt, act_fn);

            % Update personal best (Eq. 4)
            if score < pbest_scores(i)
                pbest_scores(i) = score;
                pbest_pos(i,:)  = particles(i,:);
            end
        end

        % Update global best (Eq. 5)
        [curr_best, idx] = min(pbest_scores);
        if curr_best < gbest_score
            gbest_score = curr_best;
            gbest_pos   = pbest_pos(idx,:);
        end

        fprintf(' Iteration %3d/%d | Best MAE: %.6e\n', ...
                iter, num_iterations, gbest_score);
    end

    best_intervals = gbest_pos;
    best_mae       = gbest_score;
end

% -------------------------------------------------------------------------
% extract_linear_equations: Run GP on best intervals and return equations
%   Output struct fields: slope, intercept, x_start, x_end
% -------------------------------------------------------------------------
function [equations, seg_mae] = extract_linear_equations( ...
        intervals, gp_pop, gp_depth, gp_gen, gp_opt, act_fn)

    num_segs  = length(intervals) - 1;
    equations = struct('slope', {}, 'intercept', {}, 'x_start', {}, 'x_end', {});
    seg_mae   = zeros(1, num_segs);

    symbols{1} = {'+', '*'};
    symbols{2} = {'x1', 'x2'};

    for j = 1:num_segs
        x_start = intervals(j);
        x_end   = intervals(j+1);

        X1 = linspace(x_start, x_end, 10000)';
        X  = [X1, X1];
        Y  = act_fn(X1);

        % GP-EVOLVE: evolve population (Algorithm 2)
        popu = gpols_init(gp_pop, gp_depth, symbols);
        popu = gpols_evaluate(popu, 1:gp_pop, X, Y, [], gp_opt(3:9));
        for g = 1:gp_gen
            popu = gpols_mainloop(popu, X, Y, [], gp_opt);
        end

        % GP-BESTLINEAR: extract best linear model L_j(x) = a_j*x + b_j
        [result_str, ~] = gpols_result(popu, 2);

        % Parse slope and intercept; fallback to least-squares if needed
        [a, b] = parse_linear_coefficients(result_str, X1, Y);

        % Segment MAE
        Y_approx   = a * X1 + b;
        seg_mae(j) = mean(abs(Y - Y_approx));

        % Store result
        equations(j).slope     = a;
        equations(j).intercept = b;
        equations(j).x_start   = x_start;
        equations(j).x_end     = x_end;
    end
end

% -------------------------------------------------------------------------
% parse_linear_coefficients: Extract a, b from GP result string
%   Falls back to least-squares fit if parsing fails
% -------------------------------------------------------------------------
function [a, b] = parse_linear_coefficients(result_str, X1, Y)
    coeff_match = regexp(result_str, ...
        '([+-]?\s*\d+\.?\d*(?:e[+-]?\d+)?)\s*\*\s*x1', 'tokens');
    inter_match = regexp(result_str, ...
        '\+\s*([+-]?\d+\.?\d*(?:e[+-]?\d+)?)\s*$', 'tokens');

    if ~isempty(coeff_match) && ~isempty(inter_match)
        a = str2double(strrep(coeff_match{1}{1}, ' ', ''));
        b = str2double(strrep(inter_match{1}{1}, ' ', ''));
    else
        % Fallback: least-squares linear fit
        p = polyfit(X1, Y, 1);
        a = p(1);
        b = p(2);
    end
end

% -------------------------------------------------------------------------
% compute_emax: Maximum Absolute Error over full domain
% -------------------------------------------------------------------------
function emax = compute_emax(intervals, equations, act_fn, lb, ub)
    X_full   = linspace(lb, ub, 100000)';
    Y_true   = act_fn(X_full);
    Y_approx = zeros(size(X_full));

    for j = 1:length(equations)
        mask = (X_full >= equations(j).x_start) & (X_full <= equations(j).x_end);
        Y_approx(mask) = equations(j).slope * X_full(mask) + equations(j).intercept;
    end

    emax = max(abs(Y_true - Y_approx));
end

% -------------------------------------------------------------------------
% init_particles: Initialize PSO swarm
% -------------------------------------------------------------------------
function particles = init_particles(num_particles, num_intervals, lb, ub)
    particles = zeros(num_particles, num_intervals);
    for i = 1:num_particles
        particles(i, 2:end-1) = rand(1, num_intervals-2) * (ub - lb) + lb;
        particles(i, 1)       = lb;
        particles(i, end)     = ub;
        particles(i, :)       = sort(particles(i, :));
    end
end

% -------------------------------------------------------------------------
% enforce_min_gap: Enforce minimum separation epsilon between boundaries
% -------------------------------------------------------------------------
function intervals = enforce_min_gap(intervals, lb, ub)
    epsilon        = 1e-3;
    intervals(1)   = lb;
    intervals(end) = ub;
    for j = 2:length(intervals)-1
        if intervals(j) <= intervals(j-1) + epsilon
            intervals(j) = intervals(j-1) + epsilon;
        end
    end
end

% -------------------------------------------------------------------------
% evaluate_fitness: Compute TMAE fitness (Eq. 3)
% -------------------------------------------------------------------------
function total_mae = evaluate_fitness(intervals, gp_pop, gp_depth, ...
                                       gp_gen, gp_opt, act_fn)
    num_segs  = length(intervals) - 1;
    total_mae = 0;

    symbols{1} = {'+', '*'};
    symbols{2} = {'x1', 'x2'};

    for i = 1:num_segs
        X1 = linspace(intervals(i), intervals(i+1), 10000)';
        X  = [X1, X1];
        Y  = act_fn(X1);

        popu = gpols_init(gp_pop, gp_depth, symbols);
        popu = gpols_evaluate(popu, 1:gp_pop, X, Y, [], gp_opt(3:9));
        for g = 1:gp_gen
            popu = gpols_mainloop(popu, X, Y, [], gp_opt);
        end

        [result_str, ~] = gpols_result(popu, 2);

        mse_match = regexp(result_str, ...
            'mse: (\d+\.?\d*(?:e[+-]?\d+)?)', 'tokens');
        if ~isempty(mse_match)
            seg_mae = str2double(mse_match{1}{1});
        else
            seg_mae = inf;
        end

        total_mae = total_mae + seg_mae;
    end

    total_mae = total_mae / num_segs;
end

% -------------------------------------------------------------------------
% save_results: Save intervals and linear equations to text file
% -------------------------------------------------------------------------
function save_results(filename, fn_name, num_intervals, lb, ub, ...
                       intervals, equations, seg_mae, overall_mae)
    fid = fopen(filename, 'w');
    fprintf(fid, 'PSO-GP Approximation Results\n');
    fprintf(fid, '============================\n');
    fprintf(fid, 'Activation Function : %s\n', fn_name);
    fprintf(fid, 'Number of Segments  : %d\n', num_intervals);
    fprintf(fid, 'Domain              : [%.2f, %.2f]\n', lb, ub);
    fprintf(fid, 'Overall MAE         : %.6e\n\n', overall_mae);
    fprintf(fid, 'Piecewise Linear Equations: F(x) = a*x + b\n');
    fprintf(fid, '%s\n', repmat('-', 1, 72));
    fprintf(fid, '%-5s  %-22s  %-14s  %-14s  %-12s\n', ...
            'Seg', 'Interval', 'Slope (a)', 'Intercept (b)', 'Seg MAE');
    fprintf(fid, '%s\n', repmat('-', 1, 72));
    for j = 1:length(equations)
        fprintf(fid, '%-5d  [%8.4f, %8.4f]  %+14.8f  %+14.8f  %.6e\n', ...
                j, equations(j).x_start, equations(j).x_end, ...
                equations(j).slope, equations(j).intercept, seg_mae(j));
    end
    fprintf(fid, '%s\n', repmat('-', 1, 72));
    fprintf(fid, '\nInterval Boundaries:\n');
    fprintf(fid, '%s\n', mat2str(intervals, 8));
    fclose(fid);
end
