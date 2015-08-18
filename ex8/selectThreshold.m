function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%Find the best threshold (epsilon)  based on the results from a
%   validation set (pval) and the ground truth (yval).
bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    predictions = (pval < epsilon);
    tp = sum ((predictions == yval) & (predictions == 1));
 % fp is the number of false positives: the ground truth label says it's not
 % an anomaly, but incorrectly class. it as an anomaly.
    fp = sum((predictions == 1) &(yval == 0));
%fn is the number of false negatives: the ground truth label says it's an
%anomaly, but our algorithm incorrectly classi it as not being anomalous.
    fn = sum((predictions == 0) &(yval == 1));
 
%precision and recall
prec=tp/(tp+fp);
rec=tp/(tp+fn);
   
F1=(2*prec*rec)/(prec+rec);

if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
end
end

end
