vert = 3;
N = 300000;

vl = zeros(1,2);

for i = 1:vert
    theta = 2*pi*i/vert;
    vl(i,:) = [sin(theta) cos(theta)];
end

cl = zeros(1,2);
pl = zeros(1,2);

theta1 = 2*pi*rand();
theta2 = 2*pi*rand();
seed = [0.1*sin(theta1) 0.1*cos(theta2)];

for i = 1:N
    color = ceil(rand()*vert);



    seed = (vl(color,:) - seed)./2+seed;


    pl(i,:) = seed; 

end

figure()
scatter(pl(:,1), pl(:,2),1,"Filled")