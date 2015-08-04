function idx = findClosestCentroids(X, centroids)
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

for i=1:size(X,1)
x=X(i,:);
dist=9999999999;

 for c=1:K
   cen=centroids(c,:);
   distTmp= distFind(x,cen);
   if(distTmp<=dist)
   dist=distTmp;
   idx(i,1)=c;
   end
 end    
end

end

function d = distFind(x, c)
d=sqrt(sum((x-c).^2));
end
