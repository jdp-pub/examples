l=1;
Ft = 850; 
mass = 0.025;

y0 = 0.25; 
x0 = 0.4; 
FWHM = 0.1;
sigx = FWHM/(2*sqrt(2*log(2)));

T = 3;

dx = 0.01;
dt = 0.01;

%disp(1/2/dt)
disp(1/2*sqrt(Ft/mass))

c=sqrt(Ft/mass);
dt = dx/c;
r = c*dt/dx;

ns = l/dx+1; 
pos = 0;

stry = []; 
strx = 0:dx:1+dx;
for i = 0:ns
    pos = i/ns;
    stry = [stry y0*exp(-(pos-x0)^2/(2*sigx^2))];
end


yt = [];
ft = [];
t = [];
yn = stry;

y1 = stry; %previous timestep

colorp = [1,0,0];
hold on
for i=0:dt:T*l/c+dt
    if ceil(i/dt) > 1
        y1 = yt(round(i/dt-1),:);
        stry = yt(round(i/dt),:);
    end
    for k = 2:ns-2
        yn(1,k) = 2*(1-r^2)*stry(1,k)-y1(1,k)...
            +r^2*(stry(1,k-1)+stry(1,k+1));
    end
    if mod(i,T*l/c/10) == 0
        plot(strx,stry, color = colorp)
        colorp = colorp + [-0.1,0,0.1];
    end
    yt = [yt; yn];
    ft = [ft yn(1,round(0.5*(ns+1)))];
    t = [t i];
end


xlim([0, 1])
title("Wave Propagation, Closed Boundaries")
ylabel("Amplitude (m)")
xlabel("Position (m)")
legend("$t=0$","","","","","","","","","$t=\tau$",'interpreter',"latex")
hold off
