import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import scipy
import math
from turtle import *
from scipy.ndimage import zoom
from sympy import factorint
from itertools import product
from PIL import Image
from scipy.stats import linregress
import sys


def find_nearest(value, lst):
    return min(lst, key=lambda x: abs(x - value))

def find_minima(lst):
    minima = []
    for i in range(1, len(lst) - 1):
        if lst[i] < lst[i - 1] and lst[i] < lst[i + 1]:
            minima.append((i, lst[i]))  # (index, value)
    return minima

def logistic_x():
    n = 1000
    res = 1000
    a = 4
    x = np.linspace(0,1,n)
    y = [a*xn*(1-xn) for xn in x]
    xl = [0.6]
    yl = [0]

    for xi in range(res):
        
        if xi%2==0:
            inx = np.where(x==find_nearest(xl[-1], x))[0][0]
            nv = y[inx]
            xl.append(xl[-1])
            yl.append(nv)
        else: 
            val = a*xl[-1]*(1-xl[-1])
            xl.append(val) 
            inx = np.where(x==find_nearest(xl[-1], x))[0][0]
            nv = x[inx]
            yl.append(nv)



    plt.plot(x,y) # parabola
    plt.plot(x,x) # bisector
    plt.plot(xl,yl)
    plt.show()

def pop_growth():
    P0 = 2
    Pl=[P0]
    for ni in range(20):
        r = random.random()
        P0 = (1+r)*P0
        Pl.append(P0)
        
    plt.plot(Pl)
    plt.show()

def logistic_it():
    p = (1+np.sqrt(5))/(2*np.sqrt(5))
    r = np.sqrt(5)
    n = 50
    fp = 9
    c = -1
    p1 = p
    p2 = p
    pl1 = [p]
    pl2 = [p]
    xn = (1+r)/2-r*p
    xl = [xn]
    for ni in range(n):
        #p = round(p+r*p*(1-p),fp)
        p1 = p1+r*p1*(1-p1)
        p2 = (1+r)*p2-r*p2**2
        xn = xn**2+c
        #if ni == 9:
        #    p=0.722
        #print(f"{ni}: {p}")
        pl1.append(p1)
        pl2.append(p2)
        xl.append(xn)


    print(xl)

    print([-1/r*(x-(1+r)/2) for x in xl])
    #print(pl1)
    #print()
    #print(pl2)
    stop
    #plt.plot(pl)
    #plt.show()

def quad_it():
    x = 0.5
    c = -2

    for ni in range(4):
        print(x)
        x = x**2+c
        print(x)
        print()


def collatz():
    nl = []
    nb = 1
    nf = 1.1
    na = 1000
    for A in np.linspace(nb,nf,na):
        #A = 1
        #Al = [A]
        n = 0
        while 1:
            n = n+1
            if A == 1 and n > 1:
                break

            if A % 2 == 0:
                A = A / 2
                #Al.append(A)
            else:
                A = 3*A+1
                #Al.append(A)
        nl.append(n)
    plt.plot(np.linspace(nb,nf,na), nl)
    plt.show()


def fibonacci():
    terms = 100
    a = [0,1]
    for ti in range(1,terms):
        a.append(a[ti]+a[ti-1])

    gr = (1+np.sqrt(5))/2

    er = [abs(a[ni+1]/a[ni]) for ni in range(1,len(a)-1)]
    print(er)

def chaos_game():
    n = 3
    dtheta = 2*np.pi/n
    t= np.pi/2
    tl = []
    for nx in range(n):
        tl.append(t)
        t = t+dtheta

    xl = [np.cos(theta) for theta in tl]
    yl = [np.sin(theta) for theta in tl]

    pts = []

    x0 = random.random()
    y0 = random.random()

    iterations = 10000

    for i in range(iterations):
        xi = xl.index(random.choice(xl))
        p = [xl[xi],yl[xi]]
        r = [(p[0]-x0)/2+x0, (p[1]-y0)/2+y0]
        pts.append(r)
        x0 = r[0]
        y0 = r[1]


    xp, yp = zip(*pts)


    plt.scatter(xp,yp,s=0.1)
    plt.scatter(xl,yl)
    plt.show()

def cantor(depth=10,ret=False):
    #depth = 10
    level = 0
    pts = [[0,1]]
    
    while level < depth:
        npts = []
        for inx in range(len(pts[-1])): 
            npts.append(pts[-1][inx])

            if inx%2!=0:
                continue
            npts1 = (pts[-1][inx+1]-pts[-1][inx])/3 + pts[-1][inx]
            npts2 = pts[-1][inx+1] - (pts[-1][inx+1]-pts[-1][inx])/3
            npts.append(npts1)
            npts.append(npts2)
            #npts.append(pts[-1][inx+1])
        #npts.append(1)
        pts.append(npts)
        level = level+1

    if ret == True:
        return pts

    for x in pts:
        line = []
        for inx in range(len(x)):
            if inx%2!=0:
                continue
            line.append([x[inx],x[inx+1]])
        for l in line:
            plt.hlines(pts.index(x),l[0],l[1])

    plt.show()

def triadic_num(intn):
    if intn == 0:
        return 0.0

    tn = 0
    
    # n is exponent on 3
    n = np.floor(math.log(intn,3))
    inl = []
    al = []
    if n < 0:
        inl.append(0)
        al.append(0)


    while intn > 0 and n > -10:
        if n >= 0:
            a = intn // (3**n)
            intn = intn - a*3**n

        else:
            intn = intn*3
            a = int(intn)
            intn = intn - a
            #intn = intn/3**(-n)
        #intn = intn - a*3**n
        inl.append(n)
        al.append(a)
        #print(intn)
        n = n-1
        
    

    #print(inl)
    #print(al)
    

    # construct number
    nstring = ''
    for nx in range(len(inl)):
        #print(inl[nx])
        nstring = nstring + str(int(al[nx]))
        if inl[nx] == 0:
            nstring = nstring + '.'
        
    return float(nstring)
    print(nstring)

def triad_cantor():
    pts = cantor(3,True)

    print(pts)
    #convert to base 3
    for py in pts:
        for px in py:
                    
            pts[pts.index(py)][pts[pts.index(py)].index(px)] = triadic_num(px)

    print(pts)
    for x in pts:
        line = []
        for inx in range(len(x)):
            if inx%2!=0:
                continue
            line.append([x[inx],x[inx+1]])
        for l in line:
            plt.hlines(pts.index(x),l[0],l[1])
    #print("ok")
    plt.show()

def cantor_traj():

    """
    n = 1000
    res = 1000
    a = 4
    x = np.linspace(0,1,n)
    y = [a*xn*(1-xn) for xn in x]
    xl = [0.6]
    yl = [0]

    for xi in range(res):
        
        if xi%2==0:
            inx = np.where(x==find_nearest(xl[-1], x))[0][0]
            nv = y[inx]
            xl.append(xl[-1])
            yl.append(nv)
        else: 
            val = a*xl[-1]*(1-xl[-1])
            xl.append(val) 
            inx = np.where(x==find_nearest(xl[-1], x))[0][0]
            nv = x[inx]
            yl.append(nv)



    plt.plot(x,y) # parabola
    plt.plot(x,x) # bisector
    plt.plot(xl,yl)
    plt.show()


    """



    x = cantor(3,True)[-1][5]
    print(x)
    n = 0
    nmax = 1000
    xnx = np.linspace(0,3,nmax)
    xny = np.linspace(0,3,nmax)
    x01 = np.linspace(0,1,nmax)
    xp = np.linspace(0,1.5,int(nmax/2)-1)
    xm = np.linspace(1.5,0,int(nmax/2)-1)

    yl1 = [xnx[min(range(nmax), key=lambda i: abs(xnx[i] - x))]]

    #print(xl)
    #stop

    while abs(x) < 100 and n < nmax:
        if x <= 0.5:
            x = 3*x
        else:
            x = 3*(1-x)
        yl1.append(xnx[min(range(nmax), key=lambda i: abs(xnx[i] - x))])
        n = n+1

 
    xl = [yl1[0]]
    yl = [0]


    for xi in range(nmax):
        
        if xi%2==0:
            inx = np.where(x01==find_nearest(xl[-1], x01))[0][0]
            #print(inx)
            if xl[-1] <=0.5:
                nv = xp[inx]
            else:
                nv = xm[inx-len(xm)-2]

            xl.append(xl[-1])
            yl.append(nv)
        else: 
            if xl[-1] <= 0.5:
                val = 3*xl[-1]
            else:
                val = 3*(1-xl[-1])
            xl.append(val) 
            inx = np.where(xny==find_nearest(xl[-1], xny))[0][0]
            
            nv = xny[inx]
            yl.append(val)
        if abs(xl[-1]) > 2:
            break


    print(xl)
    plt.plot(xny,xny)
    plt.plot(x01[0:len(xp)],xp)
    plt.plot(x01[len(xm)+1:-1],xm)
    plt.plot(xl,yl)
    plt.show()
    stop


def triangle_r(ax,side_length,center,depth):


    v1 = np.array([center[0], center[1]+np.sqrt(3)/6*side_length])
    v2 = np.array([center[0]-side_length/2, center[1]-np.sqrt(3)/6*side_length])
    v3 = np.array([center[0]+side_length/2, center[1]-np.sqrt(3)/6*side_length])


    
    if(depth == 0):
        triangle = patches.Polygon([v1, v2, v3], closed=True, color='black')
        ax.add_patch(triangle)
        return ax


    

    # upper triangle

    x1 = v1
    x2 = (v1+v2)/2
    x3 = (v1+v3)/2
    cr1 = (x1+x2+x3)/3

    ax=triangle_r(ax,side_length/2,cr1,depth-1)

    # Left triangle

    x1 = v2
    x2 = (v2+v1)/2
    x3 = (v2+v3)/2
    cr2 = (x1+x2+x3)/3

    ax=triangle_r(ax,side_length/2,cr2,depth-1)

    #right triangle

    x1 = v3
    x2 = (v3+v1)/2
    x3 = (v3+v2)/2
    cr3 = (x1+x2+x3)/3

    ax=triangle_r(ax,side_length/2,cr3,depth-1)

    return ax


def new_center(corner, center, d):
    direction = (center - corner) / np.linalg.norm(center - corner)
    return corner + direction * d

def square_r(ax,side_length,center,depth):
    
    diag = np.sqrt(2)/2*side_length
    
    v1 = np.array([center[0]+diag, center[1]+diag])
    v2 = np.array([center[0]+diag, center[1]-diag])
    v3 = np.array([center[0]-diag, center[1]-diag])
    v4 = np.array([center[0]-diag, center[1]+diag])
    
    if(depth == 0):
        
        square = patches.Polygon([v1, v2, v3, v4], closed=True, color='black')
        ax.add_patch(square)
        return ax

    d = 2/6*side_length
    #corners
    nvl = [new_center(v1,center,d),new_center(v2,center,d),new_center(v3,center,d),new_center(v4,center,d)]


    
    #sides
    nvl.append(center+np.array([0,side_length*np.sqrt(2)/3]))
    nvl.append(center-np.array([0,side_length*np.sqrt(2)/3]))
    nvl.append(center+np.array([side_length*np.sqrt(2)/3,0]))
    nvl.append(center-np.array([side_length*np.sqrt(2)/3,0]))
    #print(nvl)

    for c in nvl:
        ax = square_r(ax,side_length/3,c,depth-1)

    return ax

def sierpinski_gasket():
    depth = 7 

    
    fig, ax = plt.subplots()
    ax = triangle_r(ax,1,[0,0],depth)

    plt.xlim([-0.51,0.51])
    plt.ylim([-0.4,0.21])

    plt.show()

def sierpinski_carpet():
    depth = 4

    
    fig, ax = plt.subplots()
    ax = square_r(ax,1,[0,0],depth)

    plt.xlim([-1,1])
    plt.ylim([-1,1])

    plt.show()


def hilbert_curve():
    level = int(4)
    size = 200
    penup()
    goto(-size / 2.0, size / 2.0)
    pendown()
 
    # For positioning turtle
    hilbert(level, 90, size/(2**level-1))
    done()



 
 
def hilbert(level, angle, step):
 
    # Input Parameters are numeric
    # Return Value: None
    if level == 0:
        return
 
    right(angle)
    hilbert(level-1, -angle, step)
 
    forward(step)
    left(angle)
    hilbert(level-1, angle, step)
 
    forward(step)
    hilbert(level-1, angle, step)
 
    left(angle)
    forward(step)
    hilbert(level-1, -angle, step)
    right(angle)

def peano_curve():
    level = int(2)
    size = 20
    penup()
    goto(-size*10,0)
    pendown()
 
    # For positioning turtle
    peano(level, 90, size/(2**level-1))
    done()


def pbx(step,angle):
    forward(step)
    left(angle)
    forward(step)
    right(angle)
    forward(step)
    right(angle)
    forward(step)
    right(angle)
    forward(step)
    left(angle)
    forward(step)
    left(angle)
    forward(step)
    left(angle)
    forward(step)
    right(angle)
    forward(step)
 
 
def peano(level, angle, step):
 
    # Input Parameters are numeric
    # Return Value: None
    if level == 0:
        pbx(step,angle)
        return
 
    #print(heading())
    #stop
    #
    
    #pbx(step,angle)
    peano(level-1, angle, step)
    #forward(step)

    left(angle)
    #forward(step)
    peano(level-1, angle, step)
    #forward(step)
    
    #done()
    right(angle)
    peano(level-1, angle, step)
    #forward(step)


    right(angle)
    peano(level-1, angle, step)
    #forward(step)

    right(angle)
    peano(level-1, angle, step)
    #forward(step)

    left(angle)
    peano(level-1, angle, step)
    #forward(step)

    left(angle)
    peano(level-1, angle, step)
    #forward(step)

    left(angle)
    peano(level-1, angle, step)
    #forward(step)

    right(angle)
    peano(level-1, angle, step)
    #done()

def zigzag_curve(): 
    yl = []
    xl = np.linspace(0,10,1000)
    for x in xl:
        fx = x - np.floor(x)
        if fx <= 0.5:
            yl.append(2*fx)
        else:
            yl.append(2*(1-fx))

    #print(xl)

    plt.plot(xl,yl)
    plt.show()

def draw_wireframe(vertices,ax):
    # Make sure the shape is closed by adding the first vertex at the end
    closed_vertices = vertices + [vertices[0]]
    x, y = zip(*closed_vertices)
    
    ax.plot(x, y, 'k-', linewidth=0.2)  # 'k-' is black solid line

    return ax

def R(theta):
    return  np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def find_third_point(A, B, a, b,c):
    #A = np.array(A)
    #B = np.array(B)
    #c = np.linalg.norm(B - A)  # length of hypotenuse

    # Vector from A to B
    AB = B - A

    # Unit vector along AB
    AB_hat = AB / c

    # Perpendicular unit vector (rotate 90 degrees CCW)
    perp = np.array([-AB_hat[1], AB_hat[0]])

    # Use circle intersection logic
    # Distance from A along AB_hat to foot of perpendicular (point D)
    d = (b**2 - a**2 + c**2) / (2 * c)

    # Height from foot to point C using Pythagoras
    h_sq = b**2 - d**2
    if h_sq < 0:
        raise ValueError("Invalid triangle dimensions: no real solution")

    h = np.sqrt(h_sq)

    # Base point D on AB line
    D = A + d * AB_hat

    # Two possible solutions for C (above and below the hypotenuse)
    C1 = D + h * perp
    C2 = D - h * perp

    return C1

def gen_tree(ax,center,normal,lengths,depth,left=False):

    if depth == 0:
        return ax


    a = lengths[0]
    b = lengths[1]
    c = lengths[2]
    #generate square
    left = True
    if  left:
        t = a
        a = b
        b = t



    #verteces
    vs = c*normal
    v1 = R(np.pi/4) @ (vs) + center
    v2 = R(np.pi/2) @ (v1-center) + center
    v3 = R(np.pi/2) @ (v2-center) + center
    v4 = R(np.pi/2) @ (v3-center) + center


    ax = draw_wireframe([v1,v2,v3,v4],ax)


    # draw triangle

    vt = find_third_point(v1,v4,b,a,c)

    # pine shape
    ps = False
    if ps and depth % 2 == 0:
        vt = np.array([-vt[0],vt[1]])
        t = a
        a = b
        b = t


    
    ax=draw_wireframe([v1,v4,vt],ax)


    # left branch
    nt = (v1 - vt)/(np.linalg.norm(v1-vt))
    nt = np.array([nt[1],-nt[0]])
    nc = (nt)*(a/1.4)+(v1+vt)/2
    gen_tree(ax,nc,nt,[a*(a/c),b*(a/c),a],depth-1)


    # right branch
    nt = (v4 - vt)/(np.linalg.norm(v4-vt))
    nt = np.array([-nt[1],nt[0]])
    nc = (nt)*(b/1.4)+(v4+vt)/2
    gen_tree(ax,nc,nt,[a*(b/c),b*(b/c),b],depth-1)
    print(f"level: {depth}")

    return ax

    #ax.scatter(nc[0],nc[1])
    

def pythagorean_tree():
    a = 1 # left triangle side
    b = 1 # right 
    c = 1#np.sqrt(a**2+b**2) #bottom triangle side, also square side length

    depth = 10

    center = np.array([0,0])
    normal = np.array([0,1])

    fig, ax = plt.subplots()

    ax = gen_tree(ax,center,normal,[a,b,c],depth)

    plt.show()

def pythagorean_tiling():
    a = 1 # left triangle side
    b = 1 # right 
    c = 1 #np.sqrt(a**2+b**2) #bottom triangle side, also square side length

    depth = 7

    center = np.array([0,0])
    normal = np.array([0,1])

    fig, ax = plt.subplots()

    ax = gen_tree(ax,center,normal,[a,b,c],depth)


    center = np.array([0,0])
    normal = np.array([0,-1])

    fig, ax = plt.subplots()

    ax = gen_tree(ax,center,normal,[a,b,c],depth)


    plt.show()

def pi_cusanas():
    R = np.sqrt(2)/4
    r = 1/4
    p = 2/(R+r)

    n = 15

    for ni in range(n):
        p = 2/(R+r)
        r = (R+r)/2
        R = np.sqrt(R*r)
        
        print(f"n: {ni} | r_n: {r} | R_n: {R} | p_n: {p} | error: {(p-np.pi)/np.pi}")

def pi_vieta():
    n = 27 # computationally exact
    x = 1

    for ni in range(n):
        xn = np.sqrt(2)
        for nx in range(ni):
            xn = np.sqrt(2+xn)
        xn = xn/2
        x = x*xn

    print((2/x - np.pi)/np.pi)

def pi_wallis():
    n = 1001 # only works with odd numbers because of loop, error scalse as 1/n
    num = 1
    denom = 1
    x = []

    for ni in range(1,n+1):
        if ni % 2 == 0:
            num = num * ni**2
            #print(num)
        elif ni%2!=0 and ni<n:
            denom = denom * ni**2
            #print(denom)
        elif ni == n:
            denom = denom*ni**2*(ni+2)
            num = num * (ni+1)**2


    #print(num)
    #print(denom)
            #print(denom)
        #x.append(2*num/denom)
    #x = find_minima(x)
    #x = [xn[1] for xn in x]
    #print(x[-1])
    #print(2*num/denom)
    #plt.plot(x)
    #plt.show()

    #print((x[-1] - np.pi)/np.pi)
    print((2*num/denom - np.pi)/np.pi)

def pi_gregory():
    n = 100 # error scales as 1/n
    x = 0
    flip = False

    for nx in range(1,n):
        if nx%2!=0:
            if flip:
                x = x - 1/nx
                flip = False
            else: 
                x = x + 1/nx
                flip = True


    print((4*x - np.pi)/np.pi)

def pi_euler():
    d1 = 2
    d2 = 4
    d3 = 3

    n = 1000
    x1 = 0
    x2 = 0
    x3 = 0

    for nx in range(1,n):
        x1 = x1+1/nx**d1 # error scales as 1/n
        x2 = x2+1/nx**d2 # converges much quicker, 1E-10 error at n=1000

    print((6*x1)**(1/2)-np.pi)
    print((90*x2)**(1/4)-np.pi)

def pi_gauss():
    # gives 1.477, 57% error
    x = 48*np.arctan(1/48)+32*np.arctan(1/57)-20*np.arctan(1/239)
    print(np.arctan(x/np.pi))

def pi_ramanujan():
    n =3 # 3 is a correction below calculation error 1E-17
    x = 0

    for nx in range(0,n):
        x = x + np.sqrt(8)/9801*(4*np.math.factorial(nx))*(1103+26390*nx)/ \
            ((np.math.factorial(nx))**4*(396**(4*nx)))
        print(np.sqrt(8)/9801*(4*np.math.factorial(nx))*(1103+26390*nx)/ \
            ((np.math.factorial(nx))**4*(396**(4*nx))))
        #print(x)

    print((4/x - np.pi)/np.pi) # 2.3E-8 error

def pi_borwein():
    n = 4 # converged 1.55E-15 error
    x = 1/np.sqrt(2)
    y = 1/2

    for nx in range(0,n):
        x = (1-np.sqrt(1-x**2))/(1+np.sqrt(1-x**2))
        y = (1+x)**2*y-2**(nx+1)*x

    print((1/y-np.pi)/np.pi)

def pi_factor():
    n = 1000
    x = 0

    
    h = num_sfree(n)
    #print(f"{nx}: {h}")
    
    x = h/n
    

    x = np.sqrt(6/x)
    print((x - np.pi)/np.pi) # error scales about 1/n

def num_sfree(n):
    x = 0
    count = True
    #print(n)
    for nx in range(1,n):
        factors = factorint(nx)
        #print(f"{nx}: {factors}")
        count = True
        for ix in factors:
            #print(factors[ix])
            if factors[ix] > 1:
                count = False
                
                          

        if count:
            x = x+1

    #print(f"{n}: {x}")
    #print()

    return x

def pi_atan():
    x = 4*(4*np.arctan(1/5)-np.arctan(1/239))

    print((np.pi-x)/np.pi)

def rt2_it():
    x = 10

    n = 0

    while abs(abs(x)-np.sqrt(2))> 1E-14:
        x = 1/2*(x+2/x)
        n = n+1
        print(f"{n}: {x}")

def gen_w(x,y, depth):
    
    if depth ==0:
        return [x,y]


    w1 = gen_w(x/3,y/3,depth-1)
    w2 = gen_w(x/6-np.sqrt(3)*y/6+1/3,np.sqrt(3)*x/6+y/6,depth-1)
    w3 = gen_w(x/6+np.sqrt(3)*y/6+1/2,-np.sqrt(3)*x/6+y/6+np.sqrt(3)/6,depth-1)
    w4 = gen_w(x/3+2/3,y/3,depth-1)


    #print()

    xs = w1[0]
    xs = np.concatenate((xs,w2[0]))
    xs = np.concatenate((xs,w3[0]))
    xs = np.concatenate((xs,w4[0]))

    ys = w1[1]
    ys = np.concatenate((ys,w2[1]))
    ys = np.concatenate((ys,w3[1]))
    ys = np.concatenate((ys,w4[1]))



    #print(xs)

    #print(w4)
    #stop
    return [xs,ys]

def w_koch():

    depth = 9

    xs = np.linspace(0,1,100)
    ys = np.linspace(0,0,100)
    w = gen_w(xs,ys,depth)
    #stop 

    #print(w)
    #stop

    #print(w)

    x = w[0]
    y = w[1]

    #print(x)

    plt.scatter(x,y,s=0.1)
    plt.show()

    
def binary_combinations(n):
    return list(product([0, 1], repeat=n))

def binary_fraction_to_decimal(binary_str):
    """
    Convert a binary string with fractional part to a decimal number.
    
    Args:
        binary_str (str): Binary number as a string (e.g., "110.101")
    
    Returns:
        float: Decimal representation
    """
    if '.' in binary_str:
        int_part, frac_part = binary_str.split('.')
    else:
        int_part, frac_part = binary_str, ''

    # Convert integer part
    decimal = int(int_part, 2) if int_part else 0

    # Convert fractional part
    for i, digit in enumerate(frac_part):
        if digit == '1':
            decimal += 1 / (2 ** (i + 1))

    return decimal


def flip(x):
    r = random.random()
    if r > 0.5 or x == 1:
        return 1
    
    return 0

def binary_sierp():

    n = 15
    a = binary_combinations(n)
    a = [list(x) for x in a]
    b = []
    for combo in a:
        b.append([x^flip(x) for x in combo])


    w1xs = []
    w1ys = []
    w2xs = []
    w2ys = []
    w3xs = []
    w3ys = []

    for nx in range(len(a)):
        x = ""
        y = ""
        for ax in range(len(a[nx])):
            x = x+str(a[nx][ax])
            y = y+str(b[nx][ax])
        #xs.append(float(binary_fraction_to_decimal(x)))
        #ys.append(float(binary_fraction_to_decimal(y)))
        #xs.append(float(x))
        #ys.append(float(y))

        w1xs.append(float(binary_fraction_to_decimal("0.0"+x)))
        w2xs.append(float(binary_fraction_to_decimal("0.1"+x)))
        w1ys.append(float(binary_fraction_to_decimal("0.0"+y)))
        w3ys.append(float(binary_fraction_to_decimal("0.1"+y)))

        
    w3xs = w1xs
    w2ys = w1ys





    #print(w1xs)
    #print(w2ys)

    plt.scatter(w1xs,w1ys,s=0.01)
    plt.scatter(w2xs,w2ys,s=0.01)
    plt.scatter(w3xs,w3ys,s=0.01)
    plt.show()

    #print(a)
    #print()
    #print(b)

def spiral():

    n = 1000
    phi = np.linspace(0,10*np.pi,n)

    q = 0.06

    r = [q*p for p in phi] #archimedean
    #r = [np.exp(q*p) for p in phi] #logarithmic

    x = [r[nx]*np.cos(phi[nx]) for nx in range(n)]
    y = [r[nx]*np.sin(phi[nx]) for nx in range(n)]

    plt.scatter(x,y,s=0.5)
    plt.show()


def sq_spiral():
    tol = 1
    k = 0.95
    a = 200
    penup()
    goto(-300,300)
    pendown()

    while a > tol:
        forward(a)
        right(90)
        forward(a)
        a = a*k

    done()

def rn_spiral():
    tol = 1
    k = 0.9
    a = 200
    g = (1+np.sqrt(5))/(2)
    a0 = a
    penup()
    goto(-100,100)
    pendown()


    n = 1
    while a > tol:
        circle(-a,90)
        #a = a*k
        #a = a0/n
        #a = k**(n-1)*a0
        a = (1/g)**(n-1)*a0
        
        n = n+1
        

    done()

def unit32(step):
     forward(step)
     left(90)
     forward(step)
     right(90)
     forward(step)
     right(90)
     forward(2*step)
     left(90)
     forward(step)
     left(90)
     forward(step)
     right(90)
     forward(step)


def build32(step,depth):

    if(depth ==0):
        unit32(step)
        return

    build32(step,depth-1)
    left(90)
     
    build32(step,depth-1)
    right(90)
    
    build32(step,depth-1)
    right(90)

    build32(step,depth-1)
    build32(step,depth-1)
    left(90)

    build32(step,depth-1)
    left(90)

    build32(step,depth-1)
    right(90)

    build32(step,depth-1)


def curve32():
    penup()
    goto(-500,0)
    pendown()

    step = 10
    depth = 2

    build32(step,depth)

    done()


def load_png(path):
    
    # Load image and convert to grayscale (L mode = 8-bit pixels, black and white)
    img = Image.open(path).convert('L')
    matrix = np.array(img)

    #print(matrix.shape)        # (height, width)
    #print(matrix.dtype)
    return matrix


def box_dim():

    pix = load_png('wild_fractal.png')


    xl = pix.shape[0]
    yl = pix.shape[1]

    boxes = 0
    found = False
    N = []
    sl = []


    for sx in [[4,6],[8,12]]:
        boxes = 0
        bsx = xl/sx[0]
        bsy = yl/sx[1]
        #print(bsx)
        #print(bsy)
        #stop
        for boxx in range(0,sx[0]):
            for boxy in range(0,sx[1]):
                #print("boxy")
                #print(1*boxx,int(bsx*boxx))


                # indexing is not correct
                for pxx in range(int(bsx*boxx),int(bsx*boxx+bsx)):
                    for pxy in range(int(bsy*boxy),int(bsy*boxy+bsy)):
                        #print(pxx)
                        #print(pxy)
                        #print(f"last index: {int(bsy*boxy+bsy)}")
                        z = pix[pxx-1][pxy-1]
                        #print()
                        if z < 0.1:
                            #print(z)
                            found = True
                            break
                    if found:
                        break
                if found:
                    found = False
                    boxes = boxes + 1
                    #print(pxx,pxy)
                    #print(boxx,boxy)
                    #stop
        N.append(np.log(boxes))
        sl.append(np.log(sx[1]))

    print(np.exp(N))

    slope, intercept, r_value, p_value, std_err = linregress(sl, N)
    print("Slope:", slope)

    plt.plot(sl,N)
    plt.show()

def dcr(depth,s,coords):
    #devils staircase recursion

    if depth==0:
        return coords

    sx = s[0]
    sy = s[1]
    sp = s[2]
    x = coords[0]
    y = coords[1]

    yl = y - sy/2
    yr = y + sy/2

    xl = x - sx/3
    xr = x + sx/3


    sn = [1/3**sp,1/2**sp,sp+1]
    #print(depth)
    #print(sp)
    #print()
    left = dcr(depth-1,sn,[xl,yl])
    right = dcr(depth-1,sn,[xr,yr])

    return left+coords+right

def devils_staircase():
    center = [0.5,0.5]
    depth =20
    s = [1/3,1/2,2]

    pts = dcr(depth,s, center)
    xl = []
    yl = []

    #print(pts)
    print(len(pts))

    for nx in range(len(pts)):
        if nx%2==0:
            xl.append(pts[nx])
        else:
            yl.append(pts[nx])
            
    #print(xl)
    #print(yl)

    #sys.exit(0)
    #plt.scatter(xl,yl,s=0.2)
    plt.plot(xl,yl,linewidth=0.5)
    plt.show()


