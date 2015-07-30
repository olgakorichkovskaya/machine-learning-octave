function [C, sigma] = dataset3Params(X, y, Xval, yval)
C = 1;
sigma = 0.3;

steps=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sizeSteps=size(steps,1);

bestMean=999999999;

for i=1:sizeSteps
 for j=1:sizeSteps
 c=steps(i,1);
 sig=steps(j,1);
 
  model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig)); 

  predictions = svmPredict(model, Xval);
  meanTmp = mean(double(predictions ~= yval));
if(bestMean>meanTmp)
bestMean =meanTmp;
C=c;
sigma=sig;
end
    
 end
end    
  
end
