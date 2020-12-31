addpath /Applications/Dynare/4.6.3/matlab

dynare nk_with_capital noclearall nolog
Y = oo_.steady_state(1);
C = oo_.steady_state(2);
L = oo_.steady_state(3);
W = oo_.steady_state(4);
R = oo_.steady_state(5);
Pi = oo_.steady_state(6);
Q = oo_.steady_state(7);
X = oo_.steady_state(8);
RK = oo_.steady_state(9);
MC = oo_.steady_state(10);
S1 = oo_.steady_state(11);
S2 = oo_.steady_state(12);
V = oo_.steady_state(13);
M = oo_.steady_state(14);
Pstar = oo_.steady_state(15);
K = oo_.steady_state(16);
etabeta = oo_.steady_state(17);
etaL = oo_.steady_state(18);
etaA = oo_.steady_state(19);
etaR = oo_.steady_state(20);

save dynare_ss.mat Y C L W R Pi Q X RK MC S1 S2 V M Pstar K etabeta etaL etaA etaR;