% ====== simulation.m (Appendix A reconstruction) ======
%
% This script reproduces the simulation described in Chapter 5 and
% Appendix A of "Impact of an Infinite Cylinder".

close all
clear
clc

LxCGS = 160;      % Domain size in cm
lCGS = 1;         % Length of the pressure field in cm
m = 2^10;         % Number of spatial intervals
TauCGS = 0.5;     % Duration of surface forcing
TfCGS = TauCGS;   % Length of simulation
Nt = 150;         % Number of temporal intervals
rhoCGS = 1;       % Density in CGS
sigmaCGS = 70;    % Surface tension in CGS
gCGS = 980;       % Gravity in CGS

% ====== UNITS ======

L = lCGS;                 % Units of length
T = TauCGS;               % Units of time
M = rhoCGS * L^3;         % Units of mass

% ====== DIMENSIONLESS GROUPS ======

Fr = L / (T^2 * gCGS);
We = rhoCGS * L^3 / (T^2 * sigmaCGS);

% ====== NON-DIMENSIONALISE DATA ======

Lx = LxCGS / L;    % Dimensionless length of the domain
Tf = TfCGS / T;    % Dimensionless time of simulation

% ====== SPATIAL DISCRETISATION ======

dx = Lx / m;                      % Dimensionless mesh width
xVec = (-Lx/2 : dx : Lx/2)';

% ====== DtN OPERATOR ======

line = zeros(1, m + 1);
line(1) = 1 + 11/3;
line(2) = -16/9;
line(3) = -1/18;

for nn = 3:2:m
    an = -nn / (nn - 1) + (nn + 1/2) * log((nn + 1) / (nn - 1)) - 1;
    bn = -2 * nn * log((nn + 1) / (nn - 1)) + 4;
    cn = -nn / (nn + 1) + (nn - 1/2) * log((nn + 1) / (nn - 1)) - 1;

    line(nn) = line(nn) + an;
    line(nn + 1) = line(nn + 1) + bn;
    line(nn + 2) = line(nn + 2) + cn;
end

line = line / (pi * dx);

N = zeros(m + 1, m + 1);
for ii = 1:m + 1
    N(ii, :) = [fliplr(line(2:ii)), line(1:m - ii + 2)];
end

% ====== TEMPORAL DISCRETISATION ======

dt = Tf / Nt;
tVec = 0:dt:Tf;

% The appendix code constructs the centered second-difference matrix
% directly and uses it in the Crank-Nicholson block system.
DXX = diag(-2 * ones(1, m + 1)) + diag(ones(1, m), -1) + diag(ones(1, m), 1);

A = [[eye(m + 1), -dt * N / 2]; ...
     [dt * eye(m + 1) / (2 * Fr) - dt * DXX / (2 * We), eye(m + 1)]];

B = [[eye(m + 1), dt * N / 2]; ...
     [-dt * eye(m + 1) / (2 * Fr) + dt * DXX / (2 * We), eye(m + 1)]];

VecOld = zeros(2 * m + 2, 1);
Ps = zeros(m + 1, Nt + 1);

for kk = 1:length(tVec)
    Ps(:, kk) = exp(-xVec.^2) * (0.5 - 0.5 * cos(2 * pi * tVec(kk)));
end

outputDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'figures');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% ====== PRESSURE SURFACE ======

fig1 = figure('Visible', 'off');
surf(tVec, xVec, Ps, 'LineStyle', 'none', 'FaceColor', 'interp')
xlabel('t / \tau')
ylabel('x / l')
zlabel('p_s')
title('Gaussian pressure distribution')
exportgraphics(fig1, fullfile(outputDir, 'matlab_simulation_pressure_surface.png'))

% ====== SURFACE EVOLUTION ======

fig2 = figure('Visible', 'off');
plot(xVec, VecOld(1:m + 1), 'LineWidth', 2)
set(gca, 'ylim', [-0.02 0.02], 'xlim', [-Lx / 2, Lx / 2])
grid on
xlabel('x / l')
ylabel('\eta / l')
title(sprintf('t / \\tau = %.4f', tVec(1)))

snapshotSteps = [15, 55, 95, 135];
snapshotNames = { ...
    'matlab_simulation_snapshot_t15.png', ...
    'matlab_simulation_snapshot_t55.png', ...
    'matlab_simulation_snapshot_t95.png', ...
    'matlab_simulation_snapshot_t135.png'};

for kk = 1:Nt
    VecNew = A \ (B * VecOld + [zeros(m + 1, 1); -dt * (Ps(:, kk) + Ps(:, kk + 1))]);

    plot(xVec, VecNew(1:m + 1), 'LineWidth', 2)
    set(gca, 'ylim', [-0.02 0.02], 'xlim', [-Lx / 2, Lx / 2])
    grid on
    xlabel('x / l')
    ylabel('\eta / l')
    title(sprintf('t / \\tau = %.4f', tVec(kk + 1)))
    drawnow

    idx = find(snapshotSteps == kk, 1);
    if ~isempty(idx)
        exportgraphics(fig2, fullfile(outputDir, snapshotNames{idx}))
    end

    VecOld = VecNew;
end
