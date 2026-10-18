// Fixed-order adaptation of Boost Math 1.81.0 detail/bessel_ik.hpp.
// Copyright (c) 2006 Xiaogang Zhang
// Use, modification and distribution are subject to the Boost Software
// License, Version 1.0, reproduced in BOOST_LICENSE_1_0.txt.
// Temme series for x<=2; Steed continued fraction for x>2. No fitted lookup.
#ifndef OOMAO_BESSEL_FIVE_SIXTHS_HPP
#define OOMAO_BESSEL_FIVE_SIXTHS_HPP
#include <cmath>
#include <limits>

inline double oomaoBesselFiveSixths(double x) noexcept {
    constexpr double order=5.0/6.0, v=order-1.0;
    constexpr double eps=std::numeric_limits<double>::epsilon();
    constexpr double pi=3.141592653589793238462643383279502884;
    if(x<=0 || !std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    if(x<=2) {
        // Constants evaluated at the exact binary value of v, with 80-digit
        // arithmetic, then rounded once to double (see constants.json).
        constexpr double gp=0.12878702990812593, gm=-0.07228066636996079;
        constexpr double c=0.954929658551372;
        constexpr double gamma1=-0.5760165196576331, gamma2=0.9819094597711119;
        const double a=x>=std::numeric_limits<double>::min()*2 ? std::log(x/2) : std::log(x)-std::log(2.0);
        const double b=std::exp(v*a), sigma=-a*v;
        const double d=std::abs(sigma)<eps ? 1 : std::sinh(sigma)/sigma;
        double p=(gp+1)/(2*b),q=(1+gm)*b/2;
        double f=(std::cosh(sigma)*gamma1+d*(-a)*gamma2)/c;
        double coefficient=1,sum=f,sum1=p;
        for(unsigned k=1;k<256;++k) {
            f=(k*f+p+q)/(double(k)*k-v*v);
            p/=k-v;q/=k+v;
            const double h=p-k*f;
            coefficient*=x*x/(4*k);
            sum+=coefficient*f;sum1+=coefficient*h;
            if(std::abs(coefficient*f)<std::abs(sum)*eps) return 2*sum1/x;
        }
    } else {
        double a=v*v-.25,b=2*(x+1),D=1/b;
        double f=D,delta=D,previous=0,current=1,C=-a,Q=-a;
        double S=1+Q*delta;
        for(unsigned k=2;k<512;++k) {
            a-=2*(k-1);b+=2;D=1/(b+a*D);
            delta*=b*D-1;f+=delta;
            double q=(previous-(b-2)*current)/a;
            previous=current;current=q;
            C*=-a/k;Q+=C*q;S+=Q*delta;
            if(q<eps) {C*=q;previous/=q;current/=q;}
            if(std::abs(Q*delta)<std::abs(S)*eps) {
                const double K=x>=std::log(std::numeric_limits<double>::max()) ?
                    std::exp(.5*std::log(pi/(2*x))-x-std::log(S)) :
                    std::sqrt(pi/(2*x))*std::exp(-x)/S;
                return K*(.5+v+x+(v*v-.25)*f)/x;
            }
        }
    }
    return std::numeric_limits<double>::quiet_NaN(); // convergence failure
}
#endif
