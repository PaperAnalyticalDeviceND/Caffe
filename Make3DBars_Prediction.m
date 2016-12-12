A = [1 2 3 4 5 6 7 8 9 10 11 12 13];
B = [10 10 10 10 10 10 10 9 10 5 6 3 3];
C = [99.8 97.3 97.3 97.1 95.9 95.4 88.4 87.8 84.3 52.9 41.7 28.4 22.4];
D = zeros(max(A),max(B));
for i=1:length(A)
    D(A(i),B(i)) = C(i);
end
bar3(D)
set(gca,'YTickLabel',{'Rif','Iso','Eth','Amod','Tet','Dieth','AceA','CSt','Amox','Acet','CalC','Art','Ampi'});