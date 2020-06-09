function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure


figure; 




% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

hold on;
##j=0;
##k=0;
##for i = 1 : size(X,1)
##  disp(i);
##  if y(i) == 0
##    #plot(X(i,1),X(i,2),'y.','Markersize',30);
##    j+=1;
##    neg(j,:)=X(i,:);
##    disp('neg');
##  elseif y(i) == 1
##    #plot(X(i,1),X(i,2),'k+','Markersize',10);
##    k+=1;
##    pos(k,:)=X(i,:);
##    disp('pos');
##  end
##  disp('next');
##  hold on;
##end
##plot(pos(:,1),pos(:,2),'k+','Markersize',10);
##plot(neg(:,1),neg(:,2),'y.','Markersize',25);

neg = find(y==0);
pos = find(y==1);

plot(X(pos,1),X(pos,2),'k+','Markersize',10,'LineWidth',2);
plot(X(neg,1),X(neg,2),'ko','Markersize',8,'MarkerFaceColor','y');







% =========================================================================



hold off;

end
