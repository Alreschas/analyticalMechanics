#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

class Top:
    def __init__(this,Ixx,Iyy,Izz,phi,th,psi,phid,thd,psid):
        this.phi = phi
        this.th = th
        this.psi = psi
        
        this.phid = phid
        this.thd = thd
        this.psid = psid
        
        this.omex = 0
        this.omey = 0
        this.omez = 0
        
        this.Ixx = Ixx
        this.Iyy = Iyy
        this.Izz = Izz
        
        this.I = np.diag([Ixx,Iyy,Izz])
        
        this.dt = 0.0001

    #z-x-zオイラー角から、角速度ベクトルを求める
    def calcOmega(this):
        omex = this.thd * np.cos(this.psi) + this.phid * np.sin(this.psi) * np.sin(this.th)
        omey = -this.thd*np.sin(this.psi) + this.phid * np.cos(this.psi)*np.sin(this.th)
        omez = this.psid + this.phid * np.cos(this.th)
        return omex,omey,omez
 
    #角速度ベクトルを、オイラー角に変換する
    def omega2euler(this,omex,omey,omez):
        Spsi = np.sin(this.psi)
        Cpsi = np.cos(this.psi)
        Sth = np.sin(this.th)
        Cth = np.cos(this.th)
        
        ome = np.matrix([[omex],[omey],[omez]])
        m = np.matrix([[Spsi*Sth,Cpsi,0],
                       [Cpsi*Sth,-Spsi,0],
                       [Cth,0,1]])
        
        qd = np.linalg.inv(m).dot(ome)
        
        phid = qd[0,0]
        thd = qd[1,0]
        psid = qd[2,0]
        
        return phid,thd,psid
        
    #ラグランジアン
    #Ixx,Iyyが同じ時はハミルトニアンと同じ値になる
    def Lagrangian(this):
        L = 0.5*this.Ixx*(this.thd**2 + np.sin(this.th)**2 * this.phid**2) + 0.5 * this.Izz*(this.psid + this.phid * np.cos(this.th))**2
        return L
    
    #運動量
    def momentum(this):
        pphi = this.Ixx * this.phid * np.sin(this.th)**2 + this.Izz*(this.psid+this.phid*np.cos(this.th))*np.cos(this.th)
        pth = this.Ixx * this.thd
        ppsi = this.Izz*(this.psid + this.phid*np.cos(this.th))
        
#        print(pphi,pth,ppsi)
        
        return pphi,pth,ppsi
    
    def execOneStep(this):
        this.omex,this.omey,this.omez = this.calcOmega()
        omexd = (this.Iyy-this.Izz)/this.Ixx * this.omey*this.omez
        omeyd = (this.Izz-this.Ixx)/this.Iyy * this.omez*this.omex
        omezd = (this.Ixx-this.Iyy)/this.Izz * this.omex*this.omey
        
        this.omex += this.dt * omexd
        this.omey += this.dt * omeyd
        this.omez += this.dt * omezd
        
        this.phid,this.thd,this.psid = this.omega2euler(this.omex,this.omey,this.omez)
        
        this.phi += this.dt * this.phid
        this.th += this.dt * this.thd
        this.psi += this.dt * this.psid
        
        
    def plotBox(this,ax):
        Sphi = np.sin(this.phi)
        Cphi = np.cos(this.phi)
        
        Sth = np.sin(this.th)
        Cth = np.cos(this.th)
        
        Spsi = np.sin(this.psi)
        Cpsi = np.cos(this.psi)
        
        w = np.sqrt(this.Iyy**2+this.Izz**2)/4
        d = np.sqrt(this.Ixx**2+this.Izz**2)/4
        h = np.sqrt(this.Ixx**2+this.Iyy**2)/4
        points = np.matrix([
                [-w, -d, -h],
                [ w, -d, -h],
                [ w,  d, -h],
                [-w,  d, -h],
                [-w, -d,  h],
                [ w, -d,  h],
                [ w,  d,  h],
                [-w,  d,  h]]).T
    
        
        rotPhi = np.matrix([
            [Cphi,  -Sphi,  0],
            [Sphi,   Cphi,  0],
            [0,         0,  1]
            ])
    
        rotTh = np.matrix([
            [1,   0,       0],
            [0, Cth,    -Sth],
            [0, Sth,     Cth],
            ])
        
        rotPsi = np.matrix([
            [Cpsi,  -Spsi,  0],
            [Spsi,   Cpsi,  0],
            [0,         0,  1]
            ])

        points = np.array(rotPhi.dot(rotTh).dot(rotPsi).dot(points))
        
        edges = [
            [points[:,0], points[:,1], points[:,2], points[:,3]],
            [points[:,4], points[:,5], points[:,6], points[:,7]],
            [points[:,0], points[:,1], points[:,5], points[:,4]],
            [points[:,2], points[:,3], points[:,7], points[:,6]],
            [points[:,1], points[:,2], points[:,6], points[:,5]],
            [points[:,0], points[:,3], points[:,7], points[:,4]]
        ]

        faces = Poly3DCollection(edges, linewidths=1, edgecolors='r')
        faces.set_facecolor((0,0,1,0.5))
        
        ax.add_collection3d(faces)

        ome = np.matrix([[this.omex],[this.omey],[this.omez]])
        ome = rotPhi.dot(rotTh).dot(rotPsi).dot(ome)
        ax.quiver(0,0,0,ome[0],ome[1],ome[2],pivot='tail',color='r')    
                

N = 1000
ims = []


#慣性モーメント
Ixx = Iyy = 1
Izz = 2

#こまの初期値
#斜め回転
#phi0 = np.pi/2
#th0 = np.pi/2
#psi0 = np.pi/4
#phid0 = np.pi
#thd0 = 0
#psid0 = 0

#斜め回転
phi0 = np.pi/4
th0 = np.pi/4
psi0 = np.pi/4
phid0 = 0
thd0 = np.pi/4
psid0 = 0


#こまを初期化
top = Top(Ixx,Iyy,Izz,phi0,th0,psi0,phid0,thd0,psid0)

fig = plt.figure()
#ax = Axes3D(fig)
ax = fig.add_subplot(231, projection='3d')
qz2 = fig.add_subplot(232)
qp1 = fig.add_subplot(234)
qp2 = fig.add_subplot(235)
qp3 = fig.add_subplot(236)
def update(i):
    if(i == 0):
        qz2.clear()
        qp1.clear()
        qp2.clear()
        qp3.clear()
        qp1.set_xlabel('phi')
        qp1.set_ylabel('pphi')
        qp2.set_xlabel('th')
        qp2.set_ylabel('pth')
        qp3.set_xlabel('psi')
        qp3.set_ylabel('ppsi')
    ax.clear()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    for j in range(500):
        top.execOneStep()
    top.plotBox(ax)
    
    if(i%2 == 0):
        pphi,pth,ppsi = top.momentum()
        c = pth**2 + (pphi - ppsi*np.cos(top.th))**2/(np.sin(top.th)**2)
        z = (pphi - ppsi*np.cos(top.th))**2/(np.sin(top.th)**2)
        qz2.plot(top.th,z,'.b')
        qp1.plot(top.phi,pphi,'.b')
        qp2.plot(top.th,pth,'.b')
        qp3.plot(top.psi,ppsi,'.b')
        L = top.Lagrangian()
        print(L)
    
#    thst = 0
#    if(i== 0 and ppsi/pphi > 1):
#        thst = np.arccos(pphi/ppsi)
#        qz.vlines(thst,0,1)
#        
#    print(c,ppsi,pphi)

ani = animation.FuncAnimation(fig, update, interval = 50,frames=100)
#ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()