function x = emailFeatures(word_indices)

n = 1899;

x = zeros(n, 1);

for i=1:n
    index = find(word_indices==i,1);
   if(size(index,1) ~=0) 
      x(i) =1;
   end
end





% =========================================================================
    

end
