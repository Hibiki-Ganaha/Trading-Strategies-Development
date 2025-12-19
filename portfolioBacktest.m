%% Loading Data
T = readtable("Backtest Investment Strategies Using Financial Toolbox\dowPortfolio.csv");

pricesTT = table2timetable(T);

head(pricesTT);

numSample = size(pricesTT.Variables, 1);
numAssets = size(pricesTT.Variables, 2);
table(numSample, numAssets);

%% Creating Strategy Functions
function new_weights = equalWeightFcn(current_weights, pricesTT)
    % Equal-weighted portfolio allocation

    nAssets = size(pricesTT, 2);
    new_weights = ones(1,nAssets);
    new_weights = new_weights / sum(new_weights);
end

function new_weights = maxSharpeRatioFcn(current_weights, pricesTT)
    % Mean-variance portfolio allocation

    nAssets = size(pricesTT, 2);
    assetReturns = tick2ret(pricesTT);
    % Max 25% into a single asset (including cash)
    p = Portfolio('NumAssets',nAssets,...
    'LowerBound',0,'UpperBound',0.1,...
    'LowerBudget',1,'UpperBudget',1);
p = estimateAssetMoments(p, assetReturns{:,:});
new_weights = estimateMaxSharpeRatio(p);

end

function new_weights = inverseVarianceFcn(current_weights, pricesTT) 
% Inverse-variance portfolio allocation

assetReturns = tick2ret(pricesTT);
assetCov = cov(assetReturns{:,:});
new_weights = 1 ./ diag(assetCov);
new_weights = new_weights / sum(new_weights);

end

function new_weights = robustOptimFcn(current_weights, pricesTT) 
% Robust portfolio allocation

nAssets = size(pricesTT, 2);
assetReturns = tick2ret(pricesTT);

Q = cov(table2array(assetReturns));
SIGMAx = diag(diag(Q));

% Robust aversion coefficient
k = 1.1;

% Robust aversion coefficient
lambda = 0.05;

rPortfolio = mean(table2array(assetReturns))';

% Create the optimization problem
pRobust = optimproblem('Description','Robust Portfolio');

% Define the variables
% xRobust - x  allocation vector
xRobust = optimvar('x',nAssets,1,'Type','continuous','LowerBound',0.0,'UpperBound',0.1);
zRobust = optimvar('z','LowerBound',0);

% Define the budget constraint
pRobust.Constraints.budget = sum(xRobust) == 1;

% Define the robust constraint
pRobust.Constraints.robust = xRobust'*SIGMAx*xRobust - zRobust*zRobust <=0;
pRobust.Objective = -rPortfolio'*xRobust + k*zRobust + lambda*xRobust'*Q*xRobust;
x0.x = zeros(nAssets,1);
x0.z = 0;
opt = optimoptions('fmincon','Display','off');
[solRobust,~,~] = solve(pRobust,x0,'Options',opt);
new_weights = solRobust.x;

end

function new_weights = markowitzFcn(current_weights, pricesTT) 
% Robust portfolio allocation

nAssets = size(pricesTT, 2);
assetReturns = tick2ret(pricesTT);

Q = cov(table2array(assetReturns));

% Risk aversion coefficient
lambda = 0.05;

rPortfolio = mean(table2array(assetReturns))';

% Create the optimization problem
pMrkwtz = optimproblem('Description','Markowitz Mean Variance Portfolio ');

% Define the variables
% xRobust - x  allocation vector
xMrkwtz = optimvar('x',nAssets,1,'Type','continuous','LowerBound',0.0,'UpperBound',0.1);

% Define the budget constraint
pMrkwtz.Constraints.budget = sum(xMrkwtz) == 1;

% Define the Markowitz objective
pMrkwtz.Objective = -rPortfolio'*xMrkwtz + lambda*xMrkwtz'*Q*xMrkwtz;
x0.x = zeros(nAssets,1);

opt = optimoptions('quadprog','Display','off');
[solMrkwtz,~,~] = solve(pMrkwtz,x0,'Options',opt);
new_weights = solMrkwtz.x;

end

%% Compute Initial Strategy Weights
warmupPeriod = 40;

current_weights = zeros(1,numAssets);

warmupTT = pricesTT(1:warmupPeriod,:);

% Compute the initial portfolio weights for each strategy
equalWeight_initial     = equalWeightFcn(current_weights,warmupTT);
maxSharpeRatio_initial  = maxSharpeRatioFcn(current_weights,warmupTT);
inverseVariance_initial = inverseVarianceFcn(current_weights,warmupTT);
markowitz_initial       = markowitzFcn(current_weights,warmupTT);
robustOptim_initial     = robustOptimFcn(current_weights,warmupTT);

% Visualize the initial weight allocations from the strategies
strategyNames = {'Equal Weighted', 'Max Sharpe Ratio', 'Inverse Variance', 'Markowitz Optimization','Robust Optimization'};
assetSymbols = pricesTT.Properties.VariableNames;
initialWeights = [equalWeight_initial(:), maxSharpeRatio_initial(:), inverseVariance_initial(:), markowitz_initial(:), robustOptim_initial(:)];
heatmap(strategyNames, assetSymbols, initialWeights, 'title','Initial Asset Allocations','Colormap', parula);

%% Create Backtest Strategies
