function Tripods_Spacing(U,scale,times)
% Adapted from Symmlab function.  Input 'times' gives time points to add in
% spacing.

N = size(U,3);
U2 = U;
r='r'; g='g'; b='b';

count = 0;
L = length(times);
figure;
for i=0:(N-1)
    if ismember(i+1,times)
        count = count + 1;
    end
    x = 100*count + (i/(N-1));
    y=0;
    z=0;
    p1=[x,y,z]+scale*U2(1,:,i+1); 
    plot3([x+i,p1(1)+i],[y,p1(2)],[z,p1(3)],r);
    hold on
    p2=[x,y,z]+scale*U2(2,:,i+1);
    plot3([x+i,p2(1)+i],[y,p2(2)],[z,p2(3)],g);
    p3=[x,y,z]+scale*U2(3,:,i+1);
    plot3([x+i,p3(1)+i],[y,p3(2)],[z,p3(3)],b);
end 
axis equal;
set(gca,'YTickLabel',{});set(gca,'ZTickLabel',{});
set(gca,'YTick',[0 eps]);set(gca,'ZTick',[0 eps]);
view(55,11);
