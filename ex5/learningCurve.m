function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
     mm=size(1:i,2);
     x=X(1:mm, :);
     Y= y(1:mm);
     [theta1] = trainLinearReg( x, Y, lambda);

     ht=  x*theta1;
     error_train(i)=getErr(ht,Y,mm);
  
     n=size(Xval,1);
     hval= Xval*theta1;
     error_val(i) = getErr(hval,yval,n);
end
end
  
function [err] = getErr(xx,yy,mm)
    err=sum( (xx-yy).^2 );
    err=err/(2*mm);
end

