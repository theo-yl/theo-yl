%%
clear all
close all
clc

%% 1 - Model definition and data generation
% Parameters
a0 = -2.923; % True intercept
b0 = 7.18;   % True slope
sigma = 3.8; % Standard deviation of noise
N = 100;     % Number of data points

% Set random seed for reproducibility
rng(1);

% Generate input x uniformly distributed in the range [0, 50]
x = rand(N, 1) * 50; % Uniform distribution in [0, 50]

% Generate noise e
e = sigma * randn(N, 1); % Gaussian noise (mu = 0)

% Generate output y according to the relation
y = a0 + b0 * x + e;

% Plot the generated dataset
figure(1); clf;
scatter(x, y, 'filled');
title('Generated 2D Dataset');
xlabel('x');
ylabel('y');
grid on;

%% 2 - Identification using Least-Squares formula
% We will perform linear regression to estimate a0 and b0
% Prepare the design matrix H
H = [ones(N, 1), x]; % H matrix with a column of ones for intercept and x values

% Estimate parameters using the least squares method
theta = (H' * H) \ (H' * y); % Theta = (H'H)^-1 * H'y
a_hat = theta(1); % Estimated intercept
b_hat = theta(2); % Estimated slope

% Display estimated parameters
disp(['Estimated a0: ' num2str(a_hat)]);
disp(['Estimated b0: ' num2str(b_hat)]);

%% 3 - Prediction with first order polynomial
% Use estimated parameters to predict y values
y_pred = a_hat + b_hat * x;

% Plot Data vs Model prediction
figure(2); clf;
scatter(x, y, 'filled'); % Original noisy data
hold on;
plot(x, y_pred, 'r-', 'LineWidth', 2); % Predicted line
legend('Data', 'Model Prediction');
title('Data vs Model Prediction');
xlabel('x');
ylabel('y');
grid on;

%% 4 - Identification and Prediction with second order polynomial
clear all 
close all
clc

% Parameters
a0 = -2.923; 
b0 = 7.18;  
c0 = 2.8;
sigma = 12.8; 
N = 100;     

% Set random seed for reproducibility
rng(1);

% Generate input x uniformly distributed in the range [0, 50]
x = rand(N, 1) * 50; 

% Generate noise e
e = sigma * randn(N, 1); 

% Generate output y according to the polynomial relation
y = a0 + b0 * x + c0 * x.^2 + e;

% Prepare the design matrix H for a second-order polynomial
H = [ones(N, 1), x, x.^2]; 

% Estimate parameters using the least squares method
theta = (H' * H) \ (H' * y); 
a_hat = theta(1); 
b_hat = theta(2);
c_hat = theta(3);

% Display estimated parameters
disp(['Estimated a0: ' num2str(a_hat)]);
disp(['Estimated b0: ' num2str(b_hat)]);
disp(['Estimated c0: ' num2str(c_hat)]); % Fixed here

% Predict y values using the estimated parameters
y_pred = a_hat + b_hat * x + c_hat * x.^2;

% Sort x for a smoother plot
[x_sorted, sort_idx] = sort(x);
y_pred_sorted = a_hat + b_hat * x_sorted + c_hat * x_sorted.^2; % Calculate sorted predictions

% Plot the results
figure(1); clf;
scatter(x, y, 'filled'); 
hold on;
plot(x_sorted, y_pred_sorted, 'r-', 'LineWidth', 2); 
legend('Data', 'Model Prediction');
title('Data vs Model Prediction');
xlabel('x');
ylabel('y');
grid on;
