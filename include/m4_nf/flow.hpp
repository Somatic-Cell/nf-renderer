#pragma once
#include "types.hpp"

namespace m4nf {
NF_HD void hg(float dm,float dp,float a,float b,float& cdf,float& logp){
    const float s=::sqrtf(a<=b?a*a+(b-a)*dm:b*b+(a-b)*dp);
    const float lower=a*dp/(s*(b+s));
    const float upper=b*dm/(s*(a+s));
    cdf=lower<=0.5f?lower:1.0f-upper;
    if(dp==0)cdf=0;
    if(dm==0)cdf=1;
    logp=::logf(a)+::logf(b)-log_four_pi-3.0f*::logf(s);
}
NF_HD void hg_inverse(float u,float a,float b,float& dm,float& dp){
    const float v=1.0f-u,den=a*v+b*u,sa=a/den,sb=b/den;
    dp=2.0f*u*sb*sb*(a*v+u);
    dm=2.0f*v*sa*sa*(v+b*u);
}
NF_HD bool interval(const float* axis,unsigned n,float x,unsigned& lo,float& t){
    if(n<2||!finite(x)||x<axis[0]||x>axis[n-1])return false;
    unsigned l=0,r=n-1;
    while(r-l>1){unsigned m=(l+r)/2;if(x<axis[m])r=m;else l=m;}
    lo=l; t=(x-axis[l])/(axis[l+1]-axis[l]);return finite(t);
}
NF_HD float bilerp(const ModelView& m,const float* field,unsigned l,unsigned c,float s,float t){
    const unsigned j=l*m.ncos+c;
    const float low=(1.0f-t)*field[j]+t*field[j+1];
    const float high=(1.0f-t)*field[j+m.ncos]+t*field[j+m.ncos+1];
    return (1.0f-s)*low+s*high;
}
NF_HD bool physical_interval(const ModelView& m,float cosine,EndcapPolicy policy,
                             unsigned& j,float& weight){
    if(!finite(cosine)||cosine < -1.0f||cosine > 1.0f||!m.cosines||m.ncos<2)return false;
    float table_cosine=cosine;
    if(policy==EndcapPolicy::ConstantPhysical)
        table_cosine=::fminf(m.cosines[m.ncos-1],::fmaxf(m.cosines[0],cosine));
    else if(policy!=EndcapPolicy::Reject)return false;
    return interval(m.cosines,m.ncos,table_cosine,j,weight);
}
struct CrossSections {
    float scattering=0,extinction=0;
    Status status=Status::bad_input;
};
// Density/majorant/shadow tracking uses this cheap lookup. No MLP, HG or RQS.
NF_HD CrossSections cross_sections(const ModelView& m,float lambda,float cosine,
                                       EndcapPolicy policy=EndcapPolicy::Reject){
    CrossSections p;
    if(!m.wavelengths||!m.cosines||!m.scattering||!m.extinction){p.status=Status::invalid_asset;return p;}
    unsigned i,j;float wi,wj;
    if(!interval(m.wavelengths,m.nlambda,lambda,i,wi)||!physical_interval(m,cosine,policy,j,wj)){
        p.status=Status::condition_out_of_domain;return p;
    }
    p.scattering=bilerp(m,m.scattering,i,j,wi,wj);
    p.extinction=bilerp(m,m.extinction,i,j,wi,wj);
    p.status=(finite(p.scattering)&&finite(p.extinction)&&p.scattering>0&&p.extinction>=p.scattering)?
        Status::success:Status::invalid_asset;
    return p;
}
NF_HD Context prepare(const ModelView& m,float lambda,float cosine,
                          EndcapPolicy policy=EndcapPolicy::Reject){
    Context p{};p.wavelength=lambda;p.cosine=cosine;
    if(!m.weights||!m.angular_constants||!m.gs||!m.scattering||!m.extinction||!m.wavelengths||!m.cosines){p.status=Status::invalid_asset;return p;}
    unsigned i,j;float wi,wj;
    if(lambda<m.lambda_min||lambda>m.lambda_max||cosine<-1.0f||cosine>1.0f||
       !interval(m.wavelengths,m.nlambda,lambda,i,wi)||!physical_interval(m,cosine,policy,j,wj)){
        p.status=Status::condition_out_of_domain;return p;
    }
    const float g=bilerp(m,m.gs,i,j,wi,wj);
    if(!finite(g)||::fabsf(g)>=1.0f){p.status=Status::invalid_asset;return p;}
    p.a=1.0f-g;p.b=1.0f+g;
    p.external[0]=2.0f*(lambda-m.lambda_min)/(m.lambda_max-m.lambda_min)-1.0f;
    p.external[1]=cosine;
    p.strength=::sqrtf((1.0f-cosine)*(1.0f+cosine));
    p.csca=bilerp(m,m.scattering,i,j,wi,wj);
    p.cext=bilerp(m,m.extinction,i,j,wi,wj);
    // Exact HG bin-mass identity, not CDF subtraction in FP32:
    // F(mu1)-F(mu0) = (1-g)(1+g)(mu1-mu0)/(s0*s1*(s0+s1)).
    // It replaces only cancellation-prone FP64 physical preparation.
    float norm_sum=0.0f;
    for(int k=0;k<64;++k){
        const float dm0=m.angular_constants[k],dm1=m.angular_constants[k+1];
        const float dp0=m.angular_constants[65+k],dp1=m.angular_constants[65+k+1];
        const float s0=::sqrtf(p.a<=p.b?p.a*p.a+(p.b-p.a)*dm0:p.b*p.b+(p.a-p.b)*dp0);
        const float s1=::sqrtf(p.a<=p.b?p.a*p.a+(p.b-p.a)*dm1:p.b*p.b+(p.a-p.b)*dp1);
        const float mass=p.a*p.b*m.angular_constants[130+k]/(s0*s1*(s0+s1));
        if(!(mass>min_bin)||!finite(mass)){p.status=Status::invalid_spline;return p;}
        p.base_logits[k]=::logf((mass-min_bin)/(1.0f-64.0f*min_bin));
        norm_sum+=p.base_logits[k];
    }
    norm_sum*=1.0f/64.0f;
    for(int k=0;k<64;++k)p.base_logits[k]-=norm_sum;
    p.status=Status::success;return p;
}

struct NativeMlp {
    const float* weights;
    NF_HD bool operator()(int c,const float* input,float* result)const{
        float a[64]{},b[193]{};
        a[0]=input[0];a[1]=input[1];a[2]=input[2];
        for(int layer=0;layer<4;++layer){
            const int n=out_dimension(c,layer),k=in_dimension(layer);
            const float* w=weights+weight_offset(c,layer);const float* bias=w+n*k;
            for(int j=0;j<n;++j){
                float sum=0.0f;
                for(int i=0;i<k;++i)sum=::fmaf(w[j*k+i],a[i],sum);
                b[j]=sum+bias[j];if(!finite(b[j]))return false;
            }
            if(layer<3){for(int i=0;i<64;++i)a[i]=silu(b[i]);}
            else for(int i=0;i<n;++i)result[i]=b[i];
        }
        return true;
    }
};
NF_HD void probabilities(const float* logits,int n,float floor,float scale,float* out){
    float peak=logits[0];for(int k=1;k<n;++k)peak=::fmaxf(peak,logits[k]);
    float sum=0;for(int k=0;k<n;++k){out[k]=::expf(logits[k]-peak);sum+=out[k];}
    for(int k=0;k<n;++k)out[k]=floor+scale*(out[k]/sum);
}
// Accurate FP32 sum; no FP64, sorting, merging or positive-density floors.
NF_HD float sum32(const float* values,int n){
    float s=0.0f,e=0.0f;
    for(int i=0;i<n;++i){const float v=values[i]-e,t=s+v;e=(t-s)-v;s=t;}
    return s;
}
struct Spline { float x[65]{},y[65]{},d[65]{}; };
NF_HD bool knots(const float* masses,float* cumulative){
    float a[64];float peak=0;
    for(int k=0;k<64;++k){if(!(masses[k]>0)||!finite(masses[k]))return false;peak=::fmaxf(peak,masses[k]);}
    for(int k=0;k<64;++k)a[k]=masses[k]/peak;
    const float z=sum32(a,64);for(int k=0;k<64;++k)a[k]/=z;
    cumulative[0]=0;
    float sum=0.0f,error=0.0f;
    for(int k=0;k<63;++k){
        const float term=a[k]-error,t=sum+term;
        error=(t-sum)-term;sum=t;cumulative[k+1]=sum;
    }
    cumulative[64]=1;
    for(int k=0;k<64;++k)if(!(cumulative[k+1]>cumulative[k]))return false;
    return true;
}
template<class Mlp>
NF_HD bool make_spline(int layer,float retained,const Context& p,const Mlp& mlp,Spline& s){
    const bool circle=(layer%2)!=0;float x[3],v[193],w[64],h[64];
    x[0]=circle?retained:p.strength*cos_bounded(2.0f*pi*(wrap01(retained)-0.5f));
    x[1]=p.external[0];x[2]=p.external[1];
    if(!mlp(layer,x,v))return false;
    if(!circle){
        for(int k=0;k<64;++k){v[k]+=p.base_logits[k];v[64+k]+=p.base_logits[k];}
        probabilities(v,64,min_bin,1.0f-64*min_bin,w);
        probabilities(v+64,64,min_bin,1.0f-64*min_bin,h);
        for(int k=0;k<65;++k)s.d[k]=min_derivative+softplus(v[128+k]+derivative_shift);
    }else{
        probabilities(v,32,min_bin,0.5f-32*min_bin,w);
        probabilities(v+32,32,min_bin,0.5f-32*min_bin,h);
        for(int k=0;k<32;++k){w[63-k]=w[k];h[63-k]=h[k];}
        for(int k=0;k<=32;++k)s.d[k]=min_derivative+softplus(v[64+k]+derivative_shift);
        for(int k=33;k<64;++k)s.d[k]=s.d[64-k];
        s.d[64]=s.d[0];
        const float t=p.strength;
        for(int k=0;k<64;++k){w[k]=t*w[k]+(1.0f-t)/64.0f;h[k]=t*h[k]+(1.0f-t)/64.0f;}
        for(int k=0;k<65;++k)s.d[k]=t*s.d[k]+(1.0f-t);
    }
    for(int k=0;k<65;++k)if(!(s.d[k]>0)||!finite(s.d[k]))return false;
    return knots(w,s.x)&&knots(h,s.y);
}
NF_HD bool spline_map(const Spline& sp,float input,bool inverse,float& output,float& ladj){
    if(!finite(input)||input<0||input>1)return false;
    const float* lookup=inverse?sp.y:sp.x;int lo=0,hi=64;
    while(hi-lo>1){int mid=(lo+hi)/2;if(input<lookup[mid])hi=mid;else lo=mid;}
    const float x0=sp.x[lo],y0=sp.y[lo],w=sp.x[lo+1]-x0,h=sp.y[lo+1]-y0;
    const float secant=h/w,scale=::fmaxf(secant,::fmaxf(sp.d[lo],sp.d[lo+1]));
    const float s=secant/scale,d0=sp.d[lo]/scale,d1=sp.d[lo+1]/scale;
    float t;
    if(inverse){
        const float p=(input-y0)/h,c=1.0f-p,b=d0*c+p*(2.0f*s-d1),a=s-b;
        const float diff=d0*c-d1*p,root=::sqrtf(diff*diff+4.0f*s*s*p*c);
        t=b>=0?2.0f*s*p/(b+root):(root-b)/(2.0f*a);
        output=x0+w*t;
    }else{
        t=(input-x0)/w;const float c=1.0f-t;
        const float den=s+((d0-s)+(d1-s))*t*c;
        const float frac=t*(s*t+d0*c)/den,remain=c*(s*c+d1*t)/den;
        output=frac<=0.5f?y0+h*frac:y0+h-h*remain;
    }
    if(input==0)output=0;
    if(input==1)output=1;
    const float c=1.0f-t,crossv=t*c,den=s+((d0-s)+(d1-s))*crossv;
    const float num=d0*c*c+2.0f*s*crossv+d1*t*t;
    ladj=::logf(scale)+2.0f*::logf(s)+::logf(num)-2.0f*::logf(den);
    if(inverse)ladj=-ladj;
    return finite(output)&&output>=0&&output<=1&&finite(ladj);
}
template<class Mlp>
NF_HD bool flow(float& t,float& v,const Context& p,const Mlp& mlp,bool inverse,float& ladj){
    ladj=0;
    for(int k=0;k<4;++k){const int layer=inverse?3-k:k;const bool circle=layer%2;
        Spline sp; if(!make_spline(layer,circle?t:v,p,mlp,sp))return false;
        float result,ld;if(!spline_map(sp,circle?v:t,inverse,result,ld))return false;
        if(circle)v=result;else t=result;ladj+=ld;
    }
    return finite(ladj);
}
template<class Mlp>
NF_HD Result log_prob(Vec3 omega,const Context& p,const Mlp& mlp){
    Result r;if(p.status!=Status::success){r.status=p.status;return r;}
    if(!unit(omega))return r;
    omega=normalize(omega);
    const float tr=::sqrtf(omega.x*omega.x+omega.y*omega.y),z=omega.z,norm=::hypotf(tr,z);
    const float pole_offset =
    (tr / norm) * (tr / (norm + ::fabsf(z)));

    const float dm = z >= 0.0f
        ? pole_offset : 2.0f - pole_offset;

    const float dp = z >= 0.0f
        ? 2.0f - pole_offset : pole_offset;
    const float phi=tr==0 ? 0.0f : ::atan2f(omega.y,omega.x);
    float t,v=wrap01((phi+pi)/(2.0f*pi)),base,ld;
    hg(dm,dp,p.a,p.b,t,base);
    if(!flow(t,v,p,mlp,false,ld)){r.status=Status::invalid_spline;return r;}
    r.log_pdf=base+ld;r.status=finite(r.log_pdf)?Status::success:Status::nonfinite;return r;
}
template<class Mlp>
NF_HD Sample sample(float u0,float u1,const Context& p,const Mlp& mlp){
    Sample r;if(p.status!=Status::success){r.status=p.status;return r;}
    if(!finite(u0)||!finite(u1)||u0<=0||u0>=1||u1<0||u1>=1)return r;
    float t=u0,v=u1,ld,dm,dp,unused,base;
    if(!flow(t,v,p,mlp,true,ld)){r.status=Status::invalid_spline;return r;}
    hg_inverse(t,p.a,p.b,dm,dp);
    const float z=dp<=1?dp-1.0f:1.0f-dm;
    const float radius=::sqrtf(dm)*::sqrtf(dp),phi=2.0f*pi*(v-0.5f);
    const auto angle=sincos_bounded(phi);
    r.direction={radius*angle.cosine,radius*angle.sine,z};
    hg(dm,dp,p.a,p.b,unused,base);r.log_pdf=base-ld;
    r.status=(t>0&&t<1&&radius>0&&unit(r.direction)&&finite(r.log_pdf))?Status::success:Status::unrepresentable_sample;
    return r;
}
} // namespace m4nf
