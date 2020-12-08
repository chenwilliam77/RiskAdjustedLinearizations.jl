addpath /Applications/Dynare/4.6.3/matlab

beta = 0.99;
sigma = 2.0;
psi = 1.0;
eta = 1.0;
epsi = 4.45;
phi = 0.7;
rhoA = 0.9;
sA = 0.004;
rhoR = 0.7;
sR = 0.025 / 4.0;
phipi = 1.5;

save param_textbook_nk beta epsi sigma eta psi phi phipi rhoA rhoR sA sR

dynare textbook_nk noclearall nolog
