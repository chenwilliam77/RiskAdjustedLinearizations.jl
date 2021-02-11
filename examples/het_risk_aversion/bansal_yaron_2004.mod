var Y, X, SigSq, Q, V, CE, Omega, DQ, PQ, DOmega, POmega, M;

varexo ey, ex, esig;

parameters mu_y, rho_x, sig_x, rho_sig, sig_y, varsig, beta, psi, gamma;

mu_y = .0016 * 3;
rho_x = .99^3;
sig_x = sqrt((.074 * sqrt(1 - rho_x^2))^2 * 3);
rho_sig = 0.99^3;
sig_y = sqrt(.0021^2 * 3);
varsig = sqrt(.0014^2 * 3);
beta = .999^3;
psi = 2;
gamma = 9;

model;

% (1) Endowment Growth Rate
log(Y) = log(X(-1)) + sqrt(SigSq(-1)) * ey;

% (2) Long Run Risk
log(X) = rho_x * log(X(-1)) + sig_x * sqrt(SigSq(-1)) * ex;

% (3) Stochastic Volatility
SigSq = (1 - rho_sig) * sig_y^2 + rho_sig * SigSq(-1) + sqrt(SigSq(-1)) * varsig * esig;

% (4) Value Function
V = ((1 - beta) * Omega)^(1 / (1 - psi));

% (5) Certainty Equivalent
CE = ((1 - beta) / beta * (Omega - 1))^(1 / (1 - psi));

% (6) SDF
M = beta * (V(+1) / CE)^(psi - gamma) * (Y(+1) * exp(mu_y))^(-gamma);

% (7) Wealth as Consumption Claim
Omega = DOmega + POmega;

% (8) Definition of DOmega
DOmega = 1;

% (9) Definition of POmega
POmega = exp(mu_y) * M * Y(+1) * Omega(+1);

% (10) Asset Price of Endowment Tree
Q = DQ + PQ;

% (11) Dividend from Endowment
DQ = exp(mu_y) * M * Y(+1);

% (12) Capital Gains from Endowment
PQ = exp(mu_y) * M * Q(+1) * Y(+1);

end;

initval;

% Exogenous processes initialized at long-run steady state
Y = 1;
X = 1;
SigSq = sig_y^2;

% Initial guess for Omega
Omega = 1 / (1 - (beta * Y * exp(mu_y))^(1 - psi));
V = ((1 - beta) * Omega)^(1 / (1 - psi));
CE = ((1 - beta) / beta * (Omega - 1))^(1 / (1 - psi));
M = beta * (V / CE)^(psi - gamma) * (Y * exp(mu_y))^(-gamma);
DOmega = 1;
POmega = exp(mu_y) * M * Y * Omega;
Q = (exp(mu_y) * M * Y) / (1 - exp(mu_y) * M * Y);
DQ = exp(mu_y) * M * Y;
PQ = exp(mu_y) * M * Q * Y;
end;

shocks;
var ey = 1;
var ex = 1;
var esig = 1;
end;

steady;
