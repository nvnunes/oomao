// System SHA-256; byte-identical to the portable MATLAB/Java implementation.
#include "mex.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
struct Digest {
 CC_SHA256_CTX context;
 Digest(){if(!CC_SHA256_Init(&context))throw std::runtime_error("SHA-256 initialization failed.");}
 void update(const void* bytes,size_t count){const auto* p=static_cast<const uint8_t*>(bytes);while(count){CC_LONG n=static_cast<CC_LONG>(count>std::numeric_limits<CC_LONG>::max()?std::numeric_limits<CC_LONG>::max():count);if(!CC_SHA256_Update(&context,p,n))throw std::runtime_error("SHA-256 update failed.");p+=n;count-=n;}}
 void finish(unsigned char* result){if(!CC_SHA256_Final(result,&context))throw std::runtime_error("SHA-256 finalization failed.");}
};
#else
#include <dlfcn.h>
// OpenSSL EVP's stable opaque C ABI avoids a development-package dependency.
struct Crypto {
 void* library=nullptr;
 void* (*create)();void (*destroy)(void*);const void* (*sha256)();
 int (*init)(void*,const void*,void*);int (*update)(void*,const void*,size_t);int (*finish)(void*,unsigned char*,unsigned int*);
 template<class T>T symbol(const char* name){void* p=dlsym(library,name);if(!p)throw std::runtime_error(std::string("Missing OpenSSL ")+name);return reinterpret_cast<T>(p);}
 Crypto(){library=dlopen("libcrypto.so.3",RTLD_NOW|RTLD_LOCAL);if(!library)library=dlopen("libcrypto.so.1.1",RTLD_NOW|RTLD_LOCAL);if(!library)throw std::runtime_error("System OpenSSL 3 or 1.1 is required for native hashing.");
  try{create=symbol<decltype(create)>("EVP_MD_CTX_new");destroy=symbol<decltype(destroy)>("EVP_MD_CTX_free");sha256=symbol<decltype(sha256)>("EVP_sha256");init=symbol<decltype(init)>("EVP_DigestInit_ex");update=symbol<decltype(update)>("EVP_DigestUpdate");finish=symbol<decltype(finish)>("EVP_DigestFinal_ex");}catch(...){dlclose(library);throw;}}
 ~Crypto(){if(library)dlclose(library);}
};
struct Digest {
 static Crypto& api(){static Crypto value;return value;}
 void* context=nullptr;
 Digest(){auto& a=api();context=a.create();if(!context)throw std::runtime_error("OpenSSL allocation failed.");if(a.init(context,a.sha256(),nullptr)!=1){a.destroy(context);context=nullptr;throw std::runtime_error("SHA-256 initialization failed.");}}
 ~Digest(){if(context)api().destroy(context);}
 void update(const void* data,size_t bytes){if(bytes&&api().update(context,data,bytes)!=1)throw std::runtime_error("SHA-256 update failed.");}
 void finish(unsigned char* result){unsigned int count=0;if(api().finish(context,result,&count)!=1||count!=32)throw std::runtime_error("SHA-256 finalization failed.");}
};
#endif
void mexFunction(int nlhs,mxArray* out[],int nrhs,const mxArray* in[]){
 try{
  if(nlhs!=1||nrhs!=2)throw std::runtime_error("Use digest(realNumericArray,uint8Header).");auto value=in[0];auto header=in[1];
  if(!mxIsNumeric(value)||mxIsSparse(value)||mxIsComplex(value)||!mxIsUint8(header)||mxIsSparse(header)||mxIsComplex(header))throw std::runtime_error("Full real numeric array and full uint8 header required.");
  size_t count=mxGetNumberOfElements(value),width=mxGetElementSize(value);if(count>std::numeric_limits<size_t>::max()/width)throw std::runtime_error("Byte count overflows.");
  Digest digest;digest.update(mxGetData(header),mxGetNumberOfElements(header));digest.update(mxGetData(value),count*width);
  out[0]=mxCreateNumericMatrix(1,32,mxUINT8_CLASS,mxREAL);digest.finish(static_cast<unsigned char*>(mxGetData(out[0])));
 }catch(const std::exception& e){mexErrMsgIdAndTxt("oomao:digest:execution","%s",e.what());}
}
