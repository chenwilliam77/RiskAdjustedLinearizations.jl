var C W L M Q X K RK MC Pstar S1 S2 Y Pi R V etabeta etaL etaA etaR;

varexo ebeta eL eA eR;

parameters beta gamma varphi nu chi delta Xbar alpha epsi theta Pi_ss R_ss phiR phipi phiy rhobeta rhoL rhoA rhoR sigmabeta sigmaL sigmaA sigmaR;

beta = .99;
gamma = 3.8;
varphi = 1;
nu = 1;
chi = 4;
delta = 0.025;
Xbar = delta / (1 + 1 / chi);
alpha = .33;
epsi = 10;
theta = 0.7;
Pi_ss = 1.00;
R_ss = Pi_ss / beta;
phiR = 0.5;
phipi = 1.3;
phiy = .25;
rhobeta = 0.1;
rhoL = 0.1;
rhoA = 0.9;
rhoR = 0.0;
sigmabeta = 0.01;
sigmaL = 0.01;
sigmaA = 0.01;
sigmaR = 0.01;

model;

% (1) Labor supply
C^(-gamma) * W = varphi * exp(etaL) * L^nu;

% (2) SDF
M(+1) = beta * exp(etabeta(+1)) / exp(etabeta) * C(+1)^(-gamma) / C^(-gamma);

% (3) Euler equation
1 = M(+1) * R / Pi(+1);

% (4) Tobin\'s Q;
1 =  Q * Xbar^(1 / chi) * (X / K(-1))^(-1 / chi);

% (5) Return on capital (in real terms)
Q = M(+1) * (RK(+1) + Q(+1) * (1 - delta + (Xbar^(1 / chi) / (1 - 1 / chi) * (X(+1) / K)^(1 - 1 / chi) - Xbar / (chi * (chi - 1))) - (Xbar^(1 / chi) * (X(+1) / K)^(-1 / chi)) * (X(+1) / K)));

% (6) Marginal cost
MC = (1 / (1 - alpha))^(1 - alpha) * (1 / alpha)^alpha * W^(1 - alpha) * RK^alpha / exp(etaA);

% (7) Capital-labor ratio
K(-1) / L = alpha / (1 - alpha) * W / RK;

% (8) Optimal (real) reset price
Pstar = epsi / (epsi - 1) * S1 / S2;

% (9) S1 equation (numerator of optimal real reset price)
S1 = MC * Y + theta * (M(+1) * Pi(+1)^epsi * S1(+1));

% (10) S2 equation (denominator of optimal real reset price)
S2 = Y + theta * (M(+1) * Pi(+1)^(epsi - 1) * S2(+1));

% (11) Phillips Curve
Pi^(1 - epsi) = (1 - theta) * (Pstar * Pi)^(-epsi) + theta;

% (12) Price dispersion evolution
V = Pi^epsi * ((1 - theta) * (Pstar * Pi)^(-epsi) + theta * V(-1));

% (13) Taylor Rule
R / R_ss = (R(-1) / R_ss)^phiR * ((Pi / Pi_ss)^phipi * (Y / Y(-1))^phiy)^(1 - phiR) * exp(etaR);

% (14) Output market clearing
C + X = Y;

% (15) Production function
Y = exp(etaA) * K(-1)^alpha * L^(1 - alpha) / V;

% (16) Capital accumulation
K = (1 + Xbar^(1 / chi) / (1 - 1 / chi) * (X / K(-1))^(1 - 1 / chi) - Xbar / (1 - 1 / chi)) * K(-1);

% (17)-(21) Exogenous processes
etabeta = rhobeta * etabeta(-1) + sigmabeta * ebeta;
etaL = rhoL * etaL(-1) + sigmaL * eL;
etaA = rhoA * etaA(-1) + sigmaA * eA;
etaR = rhoR * etaR(-1) + sigmaR * eR;

end;

L0 = .5548;
Q0 = 1;
V0 = 1;
RK0 = 1 / beta + Xbar - 1;
FC = @(C) C + Xbar * (alpha / (1 - alpha) * varphi * L0^nu / C^(-gamma) / RK0 * L0) - ...
    (alpha / (1 - alpha))^alpha * (varphi * L0^nu / C^(-gamma) / RK0)^alpha * L0 / V0;
C0 = fsolve(FC, 2, optimoptions('fsolve', 'Display', 'none'));

M0 = beta;

initval;
etabeta = 0;
etaL = 0;
etaA = 0;
etaR = 0;
M = M0;
L = L0;
Q = Q0;
RK = RK0;
V = V0;
C = C0;
W = varphi * L^nu / C^(-gamma);
MC = (1 / (1 - alpha))^(1 - alpha) * (1 / alpha)^alpha * W^(1 - alpha) * RK^alpha;
K = alpha / (1 - alpha) * W / RK * L;
X = Xbar * K;
Y = K^alpha * L^(1 - alpha) / V;
S1 = MC * Y / (1 - theta * Pi_ss^epsi);
S2 = Y / (1 - theta * Pi_ss^(epsi - 1));
Pstar = epsi / (epsi - 1) * S1 / S2;
Pi = Pi_ss;
R = R_ss;
end;

shocks;
var ebeta = 1;
var eL = 1;
var eA = 1;
var eR = 1;
end;

steady;
