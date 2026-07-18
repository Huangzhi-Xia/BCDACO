%% Block Coordinate Descent Ant Colony Optimization (BCDACO)
function [Best_pos, Best_score, curve] = BCDACO(Np, Imax, lb, ub, dim, fobj, a, b, Q, rho, q, zeta, J, varargin)
%   J - Number of coordinate blocks (J = 3~5)
    %% ---------- Parameters ----------
    if nargin < 14 || isempty(varargin)
        groupSize = 1;
    else
        groupSize = varargin{1};
    end

    if nargin < 13 || isempty(J)
        J = min(5, ceil(dim / 5));
    end
    if nargin < 12 || isempty(zeta), zeta = 0.85; end
    if nargin < 11 || isempty(q),    q = 0.5;     end
    if nargin < 10 || isempty(rho),  rho = 0.2;   end
    if nargin < 9  || isempty(Q),    Q = 100;     end
    if nargin < 8  || isempty(b),    b = 7.0;     end
    if nargin < 7  || isempty(a),    a = 1.0;     end
    
    if ~isa(fobj, 'function_handle')
        error('fobj must be a function handle.');
    end
    
    if mod(dim, groupSize) ~= 0
        error('dim must be divisible by groupSize.');
    end

    lb = expand_bound(lb, dim, 'lb');
    ub = expand_bound(ub, dim, 'ub');

    if any(ub <= lb)
        error('Each dimension must satisfy ub > lb.');
    end

    nGroups = dim / groupSize;
    J = min(max(round(J), 1), nGroups);

    nArchive = min(Np, max(floor(q * Np), 2));

    %% ---------- Block Partition by Complete Variable Groups ----------
    blocks = make_blocks(nGroups, groupSize, J);

    %% ---------- Initialization Using Np Complete Solutions ----------
    range = ub - lb;
    population = repmat(lb, Np, 1) + rand(Np, dim) .* repmat(range, Np, 1);

    costs = zeros(Np, 1);
    for ant = 1:Np
        costs(ant) = evaluate_cost(fobj, population(ant, :));
    end

    [costs, order] = sort(costs, 'ascend');
    population = population(order, :);

    GlobalBest.Position = population(1, :);
    GlobalBest.Cost = costs(1);

    %% ---------- Block-Level Archive Initialization ----------
    BlockArchive = cell(J, 1);

    for j = 1:J
        idx = blocks{j};

        BlockArchive{j}.Position = population(1:nArchive, idx);
        BlockArchive{j}.Cost = costs(1:nArchive);

        initialQuality = relative_quality(BlockArchive{j}.Cost);
        BlockArchive{j}.Tau = (1 - rho) * ones(nArchive, 1) + Q * initialQuality;
    end

    %% ---------- Outer BCD Loop ----------
    curve = zeros(1, Imax);

    for iter = 1:Imax
        CurrentSolution = GlobalBest.Position;
        CurrentCost = GlobalBest.Cost;

        %% Cyclic Optimization of Coordinate Blocks
        for j = 1:J
            blockIdx = blocks{j};
            blockDim = numel(blockIdx);
            FixedPart = CurrentSolution;

            oldPosition = BlockArchive{j}.Position;
            oldTau = BlockArchive{j}.Tau;

            % Since the other blocks may have changed, the archived solutions must be reevaluated under the current fixed-variable context.
            oldCost = zeros(nArchive, 1);
            for i = 1:nArchive
                fullSolution = FixedPart;
                fullSolution(blockIdx) = oldPosition(i, :);
                oldCost(i) = evaluate_cost(fobj, fullSolution);
            end

            % Sort the archive using the current objective values and reorder the corresponding pheromone values accordingly.
            [oldCost, oldOrder] = sort(oldCost, 'ascend');
            oldPosition = oldPosition(oldOrder, :);
            oldTau = oldTau(oldOrder);

            %% Reference-Solution Selection Probability: tau^alpha * eta^beta
            rankIndex = (0:nArchive-1)';
            eta = exp(-0.5 * (rankIndex ./ (zeta * nArchive)).^2);

            selectionWeight = max(oldTau, realmin).^a .* max(eta, realmin).^b;
            selectionProbability = normalize_probability(selectionWeight);

            %% Generate New Solutions for the Current Block
            NewBlockSolution = zeros(Np, blockDim);
            NewBlockCost = zeros(Np, 1);

            for ant = 1:Np
                selectedIndex = roulette_select(selectionProbability);
                selectedPosition = oldPosition(selectedIndex, :);

                sigma = zeta * sum(abs(oldPosition - selectedPosition), 1) / (nArchive - 1);

                sigmaFloor = 0.01 * (ub(blockIdx) - lb(blockIdx));
                badSigma = ~isfinite(sigma) | sigma <= eps;
                sigma(badSigma) = sigmaFloor(badSigma);

                candidate = selectedPosition + sigma .* randn(1, blockDim);

                % Boundary handling
                candidate = min(max(candidate, lb(blockIdx)), ub(blockIdx));

                NewBlockSolution(ant, :) = candidate;

                fullSolution = FixedPart;
                fullSolution(blockIdx) = candidate;
                NewBlockCost(ant) = evaluate_cost(fobj, fullSolution);
            end

            %% Update the Block-Level Archive
            allPosition = [oldPosition; NewBlockSolution];
            allCost = [oldCost; NewBlockCost];

            newTauSeed = repmat(mean(oldTau), Np, 1);
            allTau = [oldTau; newTauSeed];

            [allCost, allOrder] = sort(allCost, 'ascend');
            allPosition = allPosition(allOrder, :);
            allTau = allTau(allOrder);

            keptCost = allCost(1:nArchive);
            keptPosition = allPosition(1:nArchive, :);
            keptTau = allTau(1:nArchive);

            % Archive-level pheromone update: evaporation followed by quality-based reinforcement.
            quality = relative_quality(keptCost);
            keptTau = (1 - rho) * keptTau + Q * quality;

            % Prevent nonfinite or nonpositive pheromone values during long optimization runs.
            keptTau(~isfinite(keptTau) | keptTau <= 0) = realmin;

            BlockArchive{j}.Position = keptPosition;
            BlockArchive{j}.Cost = keptCost;
            BlockArchive{j}.Tau = keptTau;

            %% Greedy Update of the Current Solution
            % Following Algorithm 1, only the newly generated ant solutions are considered for the greedy block update.
            [bestNewCost, bestAnt] = min(NewBlockCost);

            if bestNewCost < CurrentCost
                CurrentSolution(blockIdx) = NewBlockSolution(bestAnt, :);
                CurrentCost = bestNewCost;
            end
        end

        %% Update the Global Best Solution
        if CurrentCost < GlobalBest.Cost
            GlobalBest.Position = CurrentSolution;
            GlobalBest.Cost = CurrentCost;
        end

        curve(iter) = GlobalBest.Cost;

        if mod(iter, 10) == 0 || iter == 1 || iter == Imax
            meanTau = 0;
            for j = 1:J
                meanTau = meanTau + mean(BlockArchive{j}.Tau);
            end
            meanTau = meanTau / J;

            fprintf(['Iteration %d/%d: Best Cost = %.12g, ', 'Mean Archive Tau = %.6g\n'], iter, Imax, GlobalBest.Cost, meanTau);
        end
    end

    %% ---------- Outputs ----------
    Best_pos = GlobalBest.Position;
    Best_score = GlobalBest.Cost;
end

%% ========================================================================
function bound = expand_bound(bound, dim, name)
    if isscalar(bound)
        bound = repmat(bound, 1, dim);
    else
        bound = bound(:)';
        if numel(bound) ~= dim
            error('%s must be a scalar or a vector containing dim elements.', name);
        end
    end

    if any(~isfinite(bound))
        error('%s must not contain NaN or Inf.', name);
    end
end

%% ========================================================================
function blocks = make_blocks(nGroups, groupSize, J)
    % The first J - 1 blocks contain floor(nGroups / J) variable groups, while the last block contains all remaining groups.
    baseSize = floor(nGroups / J);
    blocks = cell(J, 1);

    groupStart = 1;
    for j = 1:J
        if j < J
            groupEnd = groupStart + baseSize - 1;
        else
            groupEnd = nGroups;
        end

        scalarStart = (groupStart - 1) * groupSize + 1;
        scalarEnd = groupEnd * groupSize;
        blocks{j} = scalarStart:scalarEnd;

        groupStart = groupEnd + 1;
    end
end

%% ========================================================================
function value = evaluate_cost(fobj, x)
    value = fobj(x);

    if ~isscalar(value) || ~isreal(value) || ~isfinite(value)
        error('The objective function must return a finite real scalar.');
    end
end

%% ========================================================================
function probability = normalize_probability(weight)
    weight = weight(:);
    weight(~isfinite(weight) | weight < 0) = 0;

    total = sum(weight);
    if total <= 0
        probability = ones(size(weight)) / numel(weight);
    else
        probability = weight / total;
    end
end

%% ========================================================================
function index = roulette_select(probability)
    cumulative = cumsum(probability);
    r = rand;
    index = find(r <= cumulative, 1, 'first');

    % Handle floating-point accumulation errors.
    if isempty(index)
        index = numel(probability);
    end
end

%% ========================================================================
function quality = relative_quality(cost)
    % Map objective values to (0, 1]. The best archived solution receives a quality value of 1, while the others decrease smoothly according to their relative objective-value gaps.
    cost = cost(:);
    bestCost = min(cost);
    scale = max(abs(bestCost), 1);
    gap = max(cost - bestCost, 0);

    quality = 1 ./ (1 + gap / scale);
    quality(~isfinite(quality) | quality <= 0) = realmin;
end