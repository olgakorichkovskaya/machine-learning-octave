%% K-Means Clustering
%     computeCentroids.m
%     findClosestCentroids.m
%     kMeansInitCentroids.m
clear ; close all; clc

%% ================= Part 1: Find Closest Centroids ====================
fprintf('Finding closest centroids.\n\n');
load('ex7data2.mat');

% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ===================== Part 2: Compute Means =========================
fprintf('\nComputing centroids means.\n\n');

%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');

%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% =================== Part 3: K-Means Clustering ======================
fprintf('\nRunning K-Means clustering on example dataset.\n\n');
load('ex7data2.mat');

K = 3;
max_iters = 10;

% For consistency, here we set centroids to specific values
% but in practice you want to generate them automatically, such as by
% settings them to be random examples (as can be seen in
% kMeansInitCentroids).
initial_centroids = [3 3; 6 2; 8 5];

[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ============= Part 4: K-Means Clustering on Pixels ===============
%  K-Means to compress an image. 
%  first run K-Means on the colors of the pixels in the image and
%  map each pixel on to it's closest centroid.
fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
A = double(imread('bird_small.png'));

% or   load ('bird_small.mat');

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

K = 16; 
max_iters = 10;

% When using K-Means, it is important the initialize the centroids
% randomly. 
initial_centroids = kMeansInitCentroids(X, K);

[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ================= Part 5: Image Compression ======================
%  using K-Means to compress an image. 

fprintf('\nApplying K-Means to compress an image.\n\n');

idx = findClosestCentroids(X, centroids);

X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));


fprintf('Program paused. Press enter to continue.\n');
pause;

