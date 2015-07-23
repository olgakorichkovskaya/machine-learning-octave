function p = predict(theta, X)
s=sigmoid(X*theta);
p=double(s>=0.5);
end
