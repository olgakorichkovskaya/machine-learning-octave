function centroids = computeCentroids(X, idx, K)
[m n] = size(X);

centroids = zeros(K, n);

summ=zeros(K,n);
amount=zeros(K,1);
for i=1:m
   for k=1:K
        if idx(i,1)==k
            summ(k,:)=summ(k,:)+X(i,:); 
            amount(k,1)=amount(k,1)+1;
         end 
    end
end

for k=1:K
 centroids(k,:)=summ(k,:)./amount(k,1);
end
end

