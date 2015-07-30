function sim = gaussianKernel(x1, x2, sigma)
% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

sim = 0;
% return the similarity between x1 and x2 computed using a Gaussian kernel
% with bandwidth sigma - how fast the similarity metric decreases (to 0)
 sum_=sum((x1-x2).^2);
 sim= exp(-sum_/(2*sigma^2));   
end
