% ====== DtNmatrix.m ======

close all        % Close all open figure windows (to avoid clutter)
clear            % Delete all variables from memory (fresh start)
clc              % Clear the command window (clean screen)

% ====== PARAMETERS ======

dx = 0.1;        
% Distance between grid points

m = 2^8;         
% Number of intervals (256)
% Total number of grid points will be m + 1 = 257
% The paper chose m = 256 as a practical benchmark resolution for the Gaussian DtN validation case, likely because it gives the reported accuracy in Table 4.1 and works neatly with the FFT reference.

% ====== BUILD "line" VECTOR ====== 
% This part initiates calculation (coefficients) for equation (4.46)

line = zeros(1, m+1);
% Create a row vector of size (1 x m+1), filled with zeros
% This vector will store weights used to build the DtN matrix

% ----- Near-singularity contribution (special handling) -----

% ϕ_0 from P1 and L
line(1) = 1 + 11/3; 
% ϕ_1 from L
line(2) = -16/9;
% ϕ_2 from L
line(3) = -1/18;
% These values correspond to the near-singular part: L + P1

% These values come from a special approximation near a "singular point" (where the math formula would otherwise blow up)
% They stabilize the method

% ====== FAR-FIELD CONTRIBUTIONS ======

for nn = 3:2:m
    % Loop over odd numbers: 3, 5, 7, ..., m because  those are the panel centers for Simpson-rule intervals of width 2Δx
    % This matches how the numerical integration is constructed because we use 2*delta_x as the interval

    % ---- Compute weights for three nearby points ---- 

    an = - nn/(nn - 1) + (nn + 1/2)*log((nn + 1)/(nn - 1)) - 1;
    % Weight for the "left" neighbor (f_(-1)) 

    bn = -2*nn*log((nn + 1)/(nn - 1)) + 4;
    % Weight for the "center" point (f_(0))

    cn = - nn/(nn + 1) + (nn - 1/2)*log((nn + 1)/(nn - 1)) - 1;
    % Weight for the "right" neighbor (f_1)

    % ---- Add these weights into the vector ----

    line(nn)   = line(nn)   + an;
    line(nn+1) = line(nn+1) + bn;
    line(nn+2) = line(nn+2) + cn;

    % Each group of 3 points contributes to the final operator
end

% The loop is accumulating coefficients for the far-field contribution for P2

% ====== NORMALIZATION ======

line = line / (pi * dx);
% Scale the entire vector by (π * dx)
% This comes from the mathematical formula of the integral
% Global scaling

% OVERALL THIS IS EQUATION (4.46)

% ====== BUILD MATRIX N ======

N = zeros(m+1, m+1);
% Create a square matrix (size: (m+1) x (m+1))
% This matrix will represent the DtN operator

for ii = 1:m+1

    % Extract part of the vector and flip it (mirror it)
    left_part = fliplr(line(2:ii));
    % fliplr = flip left-to-right
    % This creates the "left side" of the row
    % Shift weight to each point

    % Extract the remaining part for the right side
    right_part = line(1 : m - ii + 2);

    % Combine both parts into one full row
    N(ii, :) = [left_part, right_part];

    % Idea:
    % Each row is just a shifted version of the same pattern
    % This happens because the physics is the same everywhere (translation symmetry)
end

% ====== CREATE TEST FUNCTION ======

x = dx * (-m/2 : m/2)';
% Create grid points from -m/2 to m/2, then scale by dx
% The ' makes it a column vector

% Gaussian function
phi = exp(-x.^2);
% Define a Gaussian function: phi(x) = e^(-x^2)
% .^2 means square each element individually

% ====== APPLY DtN OPERATOR ======

phiz = N * phi;
% Multiply matrix N with vector phi
% This gives an approximation of phi_z (derivative in z-direction)

% ====== SPECTRAL METHOD (REFERENCE SOLUTION) ======

phiSpec = phi(1:end-1);
% Remove last element to match FFT size

dkx = 2*pi/(m*dx);
% Frequency spacing in Fourier space

kx = [0:m/2 -m/2+1:-1]' * dkx;
% Construct frequency vector (Fourier modes)

phizSpec = real(ifft(abs(kx) .* fft(phiSpec)));
% Steps:
% 1. fft(phiSpec): convert to frequency domain
% 2. multiply by |k| (this is DtN in Fourier space)
% 3. ifft: convert back to real space
% 4. real: remove tiny imaginary numerical noise

scriptDir = fileparts(mfilename('fullpath'));
repoDir = fileparts(scriptDir);
outputDir = fullfile(repoDir, 'results', 'figures');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% ====== PLOT 1: ORIGINAL FUNCTION ======

fig1 = figure('Visible', 'off');
plot(x, zeros(size(x)), 'k')   % Draw horizontal axis (y=0)
hold on

plot(x(1:end-1), phiSpec, 'LineWidth', 4)
% Plot Gaussian

grid on
xlabel('x')
ylabel('phi')
exportgraphics(fig1, fullfile(outputDir, 'matlab_dtn_original_function.png'))

% ====== PLOT 2: COMPARISON ======

fig2 = figure('Visible', 'off');
plot(x, zeros(size(x)), 'k')
hold on

a = plot(x(1:end-1), phizSpec, 'LineWidth', 4);
% Spectral (reference solution)

b = plot(x, phiz, 'xk', 'MarkerSize', 6, 'LineWidth', 2);
% DtN approximation (our method)

legend([a, b], 'Spectral Method', 'DtN Approximation')

grid on
xlabel('x')
ylabel('phi_z')
exportgraphics(fig2, fullfile(outputDir, 'matlab_dtn_comparison.png'))

% ====== PLOT 3: MATRIX STRUCTURE ======

fig3 = figure('Visible', 'off');
plot(N(10, :), 'LineWidth', 2)
hold on

plot(N(ceil((m+1)/4), :), 'LineWidth', 2)
plot(N(ceil((m+1)/2), :), 'LineWidth', 2)
plot(N(end-10, :), 'LineWidth', 2)

grid on
exportgraphics(fig3, fullfile(outputDir, 'matlab_dtn_matrix_structure.png'))

% This shows different rows of N
% You will see they look similar but shifted
