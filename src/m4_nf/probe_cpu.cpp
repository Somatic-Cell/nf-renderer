#include "m4_nf/asset.hpp"
#include "m4_nf/half_reference.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

template<class Mlp>
void run(const m4nf::ModelView& v,const Mlp& mlp,std::istream& in,std::ostream& out){
    unsigned op;float lambda,cosine,a,b,c;
    out<<std::setprecision(9);
    while(in>>op>>lambda>>cosine>>a>>b>>c){
        const auto ctx=m4nf::prepare(v,lambda,cosine);
        if(op==0){const auto p=m4nf::log_prob(m4nf::Vec3{a,b,c},ctx,mlp);
            out<<unsigned(p.status)<<' '<<p.log_pdf<<" 0 0 0 "<<p.log_pdf<<'\n';}
        else if(op==1){const auto s=m4nf::sample(a,b,ctx,mlp);
            const auto p=m4nf::log_prob(s.direction,ctx,mlp);
            out<<unsigned(s.status==m4nf::Status::success?p.status:s.status)<<' '<<s.log_pdf<<' '<<s.direction.x<<' '<<s.direction.y<<' '<<s.direction.z<<' '<<p.log_pdf<<'\n';}
        else throw std::runtime_error("Unsupported query op");
    }
    if(!in.eof())throw std::runtime_error("Malformed query row; expected op lambda cosine a b c");
}
int main(int argc,char**argv){
    try{
        if(argc!=5){std::cerr<<"Usage: m4_nf_probe_cpu model.bin queries.txt output.txt fp32|half_reference\n";return 2;}
        const auto asset=m4nf::load_asset(argv[1]);const auto view=asset.view();
        std::ifstream in(argv[2]);std::ofstream out(argv[3]);if(!in||!out)throw std::runtime_error("Cannot open query/output file");
        const std::string backend=argv[4];
        if(backend=="fp32")run(view,m4nf::NativeMlp{view.weights},in,out);
        else if(backend=="half_reference")run(view,m4nf::HalfReferenceMlp{view.weights},in,out);
        else throw std::runtime_error("Unknown CPU backend");
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
