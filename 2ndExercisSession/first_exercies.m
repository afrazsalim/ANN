
%The target vector.
T = [1 1; -1 -1; 1  -1]';
%Plot the boundary
plot(T(1,:),T(2,:),'*Red')
axis([-1.1 1.1 -1.1 1.1])
title('Hopfield Network State Space')
xlabel('a(1)');
ylabel('a(2)');

%Next we create a hopfield network.
net= newhop(T);
%Next we check the stability of the vector.
[Y,Pf,Af] = sim(net,3,[],T);
for i=1:20
  a = {rands(2,1)};
  [y,Pf,Af] = sim(net,{1 30},{},a);
  %Plot
   record = [cell2mat(a) cell2mat(y)];
   start = cell2mat(a);
   hold on
   plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:))
end