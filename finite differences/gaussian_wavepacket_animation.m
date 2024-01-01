 

N = 500; %space discretization
TN = 500; %time discretization
h = 1;
m = 1;
w = 10;
v=5;
tf = 1*pi;
a = 1;
dx = a/N;
dt = tf/TN;
lb = -20;
rb = 20;
s=1;
b = rb/2;
k=sqrt(w)*2;
V0 = 16/9;
k2 = k - V0;
k3 = -k;
x0 = 0;
x = linspace(a*lb, a*rb,N);
t = linspace(0,tf,TN);



%analytic

H = sparse(N,N);

p2 = -h^2/2/m;

for i=1:N
    H(i,i) = p2*(-2/dx^2);
    if(i ~= 1)
        H(i,i-1) = p2/dx^2;
    end
    if(i ~= N)
        H(i,i+1) = p2/dx^2;
    end
end

[R,D] = eigs(H,N);
psit = zeros(N,1);
psi_i = psitev(x,0,w,s,v,k);
psit(:,1) = psi_i; %set initial time

%finite differences time evolution
for tn = 1:TN
    if(tn == 1)
        psit(:,2) = dt/i/2/h.*H*(psitev(x,t(tn),w,s,v,k))./sqrt(2);
        continue
    end

    psit(:,tn+1) = (dt/i/2/h.*H*(psitev(x-x0,t(tn),w,s,v,k)+...
                                psitevR(x-x0,t(tn),w,s,v,k3,V0,b)+...
                                psitevT(x-x0,t(tn),w,s,v,k2,V0,b))./sqrt(2)+...
                                ...
                                (psitev(x-x0,t(tn-1),w,s,v,k)+...
                                psitevR(x-x0,t(tn-1),w,s,v,k3,V0,b)+...
                                psitevT(x-x0,t(tn-1),w,s,v,k2,V0,b))./sqrt(2));
     
     
    if(sum(psit(:,tn+1) > 1.2))
        psit(:,tn+1)=normalize(psit(:,tn+1),dx);
    end
end


close all;
figure()
for i=1:TN
    plot3(x,imag(psit(:,i)),real(psit(:,i)), "LineWidth", 2);
    %plot(abs(psit(:,i)).^2, "LineWidth", 2);
    xlabel("x")

    ylabel("imagniary")
    zlabel("real")

    ylim([-1,1]);
    xlim([a*lb, a*rb]);
    zlim([-2,2])
    grid on;
    drawnow
end

hold off

function vr = V(x,m,w)
    vr = 1/2*m*w^2*x^2;
end

function norma = normalize(M,dx)
    A = 1/sqrt(dx*sum(abs(M).^2));
    norma = M*A;
end

function psi = psitev(x,t,w,s,v,k)
    psi = 1*exp(-(x-v*t).^2.*(1-i*k*t)./(2+t^2/2))./(2+i*t^2/2);
    psi = transpose(psi);
end

function psi = psitevR(x,t,w,s,v,k,V0,b)
    R = 0;%1-(1/(1+sin(-k-V0)^2/V0));
    psi = R*exp(-(x+v*t).^2.*(1-i*k*t)./(2+t^2/2))./(2+i*t^2/2);
    psi = transpose(psi);
end

function psi = psitevT(x,t,w,s,v,k,V0,b)
    T = 0;%(1/(1+sin(k)^2/V0));
    psi = -T*exp(-(x-v*t).^2.*(1-i*k*t)./(2+t^2/2))./(2+i*t^2/2);
    psi = transpose(psi);
end