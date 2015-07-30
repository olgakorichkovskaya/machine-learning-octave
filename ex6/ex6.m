clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================
fprintf('Loading and Visualizing Data ...\n')
load('ex6data1.mat');
%plotData(X, y);

%% ==================== Part 1: Training Linear SVM ====================
%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.

load('ex6data1.mat');
fprintf('\nTraining Linear SVM ...\n')

% change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 0.10;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
%visualizeBoundaryLinear(X, y, model);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% =============== Part 2: Implementing Gaussian Kernel ===============
%fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% =============== Part 4: Visualizing the next Dataset 2 ================
fprintf('Loading and Visualizing Data ...\n')

load('ex6data2.mat');
plotData(X, y);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
load('ex6data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% =============== Part 6: Visualizing Dataset 3 ================
fprintf('Loading and Visualizing Data ...\n')
load('ex6data3.mat');
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
load('ex6data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

