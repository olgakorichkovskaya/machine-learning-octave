function plotData2Dim(x,x1, y,x_label,x1_label,y_label)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

plot(x, y, x1,y ); % Plot the data
ylabel(y_label); % Set the y􀀀axis label
xlabel(x_label); % Set the x􀀀axis label
%plot(x1, y, 'rx', 'MarkerSize', 2); % Plot the data
figure; % open a new figure window
end
