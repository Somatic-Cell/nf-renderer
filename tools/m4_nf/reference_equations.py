"""Independent PyTorch replay of the fetched training inference equations.

This is NOT a teacher-data implementation and is NOT a replacement for the
training project's modules. validate.py --reference training runs the actual
ConditionalSphereFlow instead. This replay allows local algebra checks when
Zuko/the original Python environment is unavailable.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F

def hg(dm,dp,a,b):
    s=torch.where(a<=b,a.square()+(b-a)*dm,b.square()+(a-b)*dp).sqrt()
    lower=a*dp/(s*(b+s));upper=b*dm/(s*(a+s))
    t=torch.where(lower<=.5,lower,1-upper)
    t=torch.where(dp==0,0,t);t=torch.where(dm==0,1,t)
    return t,a.log()+b.log()-math.log(4*math.pi)-3*s.log()

def hg_inv(u,a,b):
    c=1-u;den=a*c+b*u
    return 2*c*(a/den).square()*(c+b*u),2*u*(b/den).square()*(a*c+u)

def prepare(c,g):
    c=torch.as_tensor(c,dtype=torch.float64);g=torch.as_tensor(g,dtype=torch.float64)
    a,b=1-g,1+g
    half=torch.arange(65,dtype=c.dtype)*math.pi/128
    t,_=hg(2*half.cos().square(),2*half.sin().square(),a[:,None],b[:,None])
    t[:,0]=0;t[:,-1]=1
    masses=t.diff(dim=-1)
    if not bool((masses>1e-5).all()):raise ValueError('invalid physical knots')
    logits=((masses-1e-5)/(1-64e-5)).log()
    logits-=logits.mean(-1,keepdim=True)
    ext=torch.stack((2*(c[:,0]-380)/450-1,c[:,1]),-1).float()
    return ext,a.float(),b.float(),((1-c[:,1])*(1+c[:,1])).sqrt().float(),logits.float()

def parameters(state,layer,retained,ctx):
    ext,a,b,strength,base=ctx;circle=layer%2
    feature=retained if circle else strength*(2*math.pi*(retained.remainder(1)-.5)).cos()
    x=torch.cat((feature[:,None],ext),-1)
    for l in range(4):
        key=f'couplings.{layer}.hyper.{2*l}'
        x=F.linear(x,state[key+'.weight'],state[key+'.bias'])
        if l<3:x=F.silu(x)
    sizes=(32,32,33) if circle else (64,64,65)
    w,h,d=x.split(sizes,-1)
    d=1e-4+F.softplus(d+math.log(math.expm1(1-1e-4)))
    if not circle:
        w=1e-5+(1-64e-5)*(w+base).softmax(-1)
        h=1e-5+(1-64e-5)*(h+base).softmax(-1)
    else:
        w=1e-5+(.5-32e-5)*w.softmax(-1)
        h=1e-5+(.5-32e-5)*h.softmax(-1)
        w=torch.cat((w,w.flip(-1)),-1);h=torch.cat((h,h.flip(-1)),-1)
        d=torch.cat((d,d[:,1:-1].flip(-1)),-1)
        s=strength[:,None]
        w=s*w+(1-s)/64;h=s*h+(1-s)/64;d=s*d+(1-s)
        d=torch.cat((d,d[:,:1]),-1)
    w=w/w.amax(-1,keepdim=True);h=h/h.amax(-1,keepdim=True)
    w=w/w.sum(-1,keepdim=True);h=h/h.sum(-1,keepdim=True)
    zeros=torch.zeros_like(w[:,:1]);ones=torch.ones_like(zeros)
    horizontal=torch.cat((zeros,w.cumsum(-1)[:,:-1],ones),-1)
    vertical=torch.cat((zeros,h.cumsum(-1)[:,:-1],ones),-1)
    return horizontal,vertical,d

def transform(value,x,y,d,inverse):
    lookup=y if inverse else x
    index=(torch.searchsorted(lookup.contiguous(),value[:,None].contiguous(),right=True).squeeze(-1)-1).clamp(0,63)
    def get(t,shift=0):return t.gather(-1,(index+shift)[:,None]).squeeze(-1)
    x0,y0,w,h,d0,d1=get(x),get(y),get(x.diff(dim=-1)),get(y.diff(dim=-1)),get(d),get(d,1)
    s=h/w;scale=torch.maximum(s,torch.maximum(d0,d1));s=s/scale;d0=d0/scale;d1=d1/scale
    if inverse:
        p=(value-y0)/h;c=1-p;b=d0*c+p*(2*s-d1);a=s-b
        difference=d0*c-d1*p
        root=(difference.square()+4*s.square()*p*c).sqrt()
        positive=b>=0
        den_a=torch.where(positive,1,2*a);den_b=torch.where(positive,b+root,1)
        t=torch.where(positive,2*s*p/den_b,(root-b)/den_a)
        out=x0+w*t
    else:
        t=(value-x0)/w;c=1-t
        den=s+((d0-s)+(d1-s))*t*c
        frac=t*(s*t+d0*c)/den;rem=c*(s*c+d1*t)/den
        out=torch.where(frac<=.5,y0+h*frac,y0+h-h*rem)
    out=torch.where(value==0,0,out);out=torch.where(value==1,1,out)
    c=1-t;cross=t*c
    denominator=s+((d0-s)+(d1-s))*cross
    numerator=d0*c.square()+2*s*cross+d1*t.square()
    ld=scale.log()+2*s.log()+numerator.log()-2*denominator.log()
    return out,-ld if inverse else ld

def flow(state,value,ctx,inverse):
    ld=torch.zeros_like(value[:,0])
    for layer in (range(3,-1,-1) if inverse else range(4)):
        changed=layer%2
        x,y,d=parameters(state,layer,value[:,1-changed],ctx)
        output,ladj=transform(value[:,changed],x,y,d,inverse)
        value=torch.stack((value[:,0],output),-1) if changed else torch.stack((output,value[:,1]),-1)
        ld+=ladj
    return value,ld

class EquationReplay:
    def __init__(self,state):self.state=state
    def log_prob(self,direction,ctx):
        x=torch.as_tensor(direction,dtype=torch.float32);x=x/torch.linalg.vector_norm(x,dim=-1)[:,None]
        tr=torch.linalg.vector_norm(x[:,:2],dim=-1);z=x[:,2];norm=torch.hypot(tr,z)
        near=(tr/norm)*(tr/(norm+z.abs()))
        dm=torch.where(z>=0,near,2-near);dp=torch.where(z>=0,2-near,near)
        phi=torch.atan2(x[:,1],x[:,0]);phi=torch.where(tr==0,0,phi)
        v=((phi+math.pi)/(2*math.pi)).remainder(1)
        t,loghg=hg(dm,dp,ctx[1],ctx[2])
        _,ld=flow(self.state,torch.stack((t,v),-1),ctx,False)
        return loghg+ld
    def sample_from_uniform(self,u,ctx):
        value,ld=flow(self.state,torch.as_tensor(u,dtype=torch.float32),ctx,True)
        dm,dp=hg_inv(value[:,0],ctx[1],ctx[2])
        mu=torch.where(dp<=1,dp-1,1-dm);radius=dm.sqrt()*dp.sqrt()
        phi=2*math.pi*(value[:,1]-.5)
        xyz=torch.stack((radius*phi.cos(),radius*phi.sin(),mu),-1)
        _,base=hg(dm,dp,ctx[1],ctx[2])
        return xyz,base-ld
