addpath /Applications/Dynare/4.6.3/matlab

dynare bansal_yaron_2004 noclearall nolog
Y = oo_.steady_state(1);
X = oo_.steady_state(2);
SigSq = oo_.steady_state(3);
Q = oo_.steady_state(4);
V = oo_.steady_state(5);
CE = oo_.steady_state(6);
Omega = oo_.steady_state(7);
DQ = oo_.steady_state(8);
PQ = oo_.steady_state(9);
DOmega = oo_.steady_state(10);
POmega = oo_.steady_state(11);
M = oo_.steady_state(12);

save dynare_ss.mat Y X SigSq Q V CE Omega DQ PQ DOmega POmega M;