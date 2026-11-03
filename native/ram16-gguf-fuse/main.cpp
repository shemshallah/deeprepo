#define FUSE_USE_VERSION 35
#include <fuse3/fuse.h>
#include <curl/curl.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <zlib.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

using boost::property_tree::ptree;
namespace {
constexpr uint64_t kVirtualSize = 32ull * 1024 * 1024 * 1024;
constexpr size_t kPageSize = 4u * 1024u * 1024u;
const std::vector<uint8_t> kZeroPage(kPageSize, 0);

std::string env(const char *name, const std::string &fallback = {}) {
    const char *v = std::getenv(name); return v && *v ? v : fallback;
}
std::string hex(const unsigned char *p, size_t n) {
    static const char *d = "0123456789abcdef"; std::string o; o.resize(n * 2);
    for (size_t i=0;i<n;i++){o[2*i]=d[p[i]>>4];o[2*i+1]=d[p[i]&15];} return o;
}
std::string sha256(const std::vector<uint8_t> &b) {
    unsigned char h[SHA256_DIGEST_LENGTH]; SHA256(b.data(), b.size(), h); return hex(h, sizeof(h));
}
std::string sha256s(const std::string &s) {
    unsigned char h[SHA256_DIGEST_LENGTH]; SHA256(reinterpret_cast<const unsigned char*>(s.data()), s.size(), h); return hex(h, sizeof(h));
}
std::vector<unsigned char> hmac256(const std::vector<unsigned char> &key, const std::string &data) {
    unsigned char out[EVP_MAX_MD_SIZE]; unsigned int n=0;
    if (!HMAC(EVP_sha256(), key.data(), (int)key.size(), reinterpret_cast<const unsigned char*>(data.data()), data.size(), out, &n))
        throw std::runtime_error("OpenSSL HMAC failed");
    return {out, out+n};
}
std::string utc_date(const char *fmt) {
    std::time_t t=std::time(nullptr); std::tm tm{}; gmtime_r(&t,&tm); char b[32];
    std::strftime(b,sizeof(b),fmt,&tm); return b;
}
std::string url_encode_path(const std::string &s) {
    static const char *h="0123456789ABCDEF"; std::string o;
    for (unsigned char c:s) {
        if ((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='-'||c=='_'||c=='.'||c=='~'||c=='/') o.push_back((char)c);
        else {o.push_back('%');o.push_back(h[c>>4]);o.push_back(h[c&15]);}
    } return o;
}
struct CurlBuffer { std::vector<uint8_t> bytes; };
size_t curl_write(char *p,size_t a,size_t b,void *u) {
    size_t n=a*b; auto *x=static_cast<CurlBuffer*>(u); x->bytes.insert(x->bytes.end(),p,p+n); return n;
}

class B2S3 {
    std::string key_id_, secret_, bucket_, endpoint_, scheme_, host_, region_;
public:
    B2S3() {
        key_id_=env("B2_READ_KEY_ID",env("B2_KEY_ID")); secret_=env("B2_READ_APPLICATION_KEY",env("B2_APPLICATION_KEY")); bucket_=env("B2_BUCKET"); endpoint_=env("B2_ENDPOINT");
        if(key_id_.empty()||secret_.empty()||bucket_.empty()||endpoint_.empty()) throw std::runtime_error("B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET and B2_ENDPOINT are required");
        auto p=endpoint_.find("://"); scheme_=p==std::string::npos?"https":endpoint_.substr(0,p); host_=p==std::string::npos?endpoint_:endpoint_.substr(p+3);
        while(!host_.empty()&&host_.back()=='/') host_.pop_back();
        auto slash=host_.find('/'); if(slash!=std::string::npos) host_=host_.substr(0,slash);
        auto s=host_.find("s3."); auto e=host_.find(".backblazeb2.com");
        if(!env("B2_REGION").empty()) region_=env("B2_REGION");
        else { if(s==std::string::npos||e==std::string::npos||e<=s+3) throw std::runtime_error("B2_ENDPOINT must be a regional S3 endpoint such as https://s3.us-west-004.backblazeb2.com"); region_=host_.substr(s+3,e-(s+3)); }
        if(curl_global_init(CURL_GLOBAL_DEFAULT)!=CURLE_OK) throw std::runtime_error("curl initialization failed");
    }
    std::vector<uint8_t> get(const std::string &key) const {
        const std::string path="/"+bucket_+"/"+key;
        const std::string uri=url_encode_path(path), url=scheme_+"://"+host_+uri;
        const std::string date=utc_date("%Y%m%dT%H%M%SZ"), day=date.substr(0,8), payload=sha256s("");
        const std::string canonical_headers="host:"+host_+"\n"+"x-amz-content-sha256:"+payload+"\n"+"x-amz-date:"+date+"\n";
        const std::string signed_headers="host;x-amz-content-sha256;x-amz-date";
        const std::string canonical="GET\n"+uri+"\n\n"+canonical_headers+"\n"+signed_headers+"\n"+payload;
        const std::string scope=day+"/"+region_+"/s3/aws4_request";
        const std::string to_sign="AWS4-HMAC-SHA256\n"+date+"\n"+scope+"\n"+sha256s(canonical);
        std::vector<unsigned char> k0={'A','W','S','4'}; k0.insert(k0.end(),secret_.begin(),secret_.end());
        auto kd=hmac256(k0,day), kr=hmac256(kd,region_), ks=hmac256(kr,"s3"), signing=hmac256(ks,"aws4_request");
        const std::string sig=hex(hmac256(signing,to_sign).data(),32);
        const std::string auth="Authorization: AWS4-HMAC-SHA256 Credential="+key_id_+"/"+scope+", SignedHeaders="+signed_headers+", Signature="+sig;
        for(int attempt=0;attempt<5;attempt++) {
            CurlBuffer out;
            static thread_local std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
            CURL *c=curl.get(); if(!c) throw std::runtime_error("curl_easy_init failed");
            curl_easy_reset(c); // retains the per-thread connection/DNS/TLS caches
            struct curl_slist *request_headers=nullptr;
            request_headers=curl_slist_append(request_headers,auth.c_str());
            const std::string date_header="x-amz-date: "+date, hash_header="x-amz-content-sha256: "+payload;
            request_headers=curl_slist_append(request_headers,date_header.c_str());
            request_headers=curl_slist_append(request_headers,hash_header.c_str());
            curl_easy_setopt(c,CURLOPT_URL,url.c_str()); curl_easy_setopt(c,CURLOPT_HTTPHEADER,request_headers);
            curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,curl_write); curl_easy_setopt(c,CURLOPT_WRITEDATA,&out);
            curl_easy_setopt(c,CURLOPT_FOLLOWLOCATION,0L); curl_easy_setopt(c,CURLOPT_CONNECTTIMEOUT,20L); curl_easy_setopt(c,CURLOPT_TIMEOUT,180L);
            curl_easy_setopt(c,CURLOPT_NOSIGNAL,1L); curl_easy_setopt(c,CURLOPT_USERAGENT,"ram16-gguf-fuse/1.0");
            CURLcode rc=curl_easy_perform(c); long status=0; curl_easy_getinfo(c,CURLINFO_RESPONSE_CODE,&status);
            curl_slist_free_all(request_headers);
            if(rc==CURLE_OK&&status>=200&&status<300) return out.bytes;
            if(rc==CURLE_OK&&status==404) throw std::runtime_error("B2 object not found: "+key);
            const bool retry_http=(status==408||status==429||status==500||status==502||status==503||status==504);
            if(attempt==4||(!retry_http&&rc==CURLE_OK)) {
                if(rc!=CURLE_OK) throw std::runtime_error(std::string("B2 HTTP read failed: ")+curl_easy_strerror(rc));
                throw std::runtime_error("B2 S3 GET HTTP "+std::to_string(status)+" for "+key+": "+std::string(out.bytes.begin(),out.bytes.end()).substr(0,512));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250*(1<<attempt)));
        }
        throw std::runtime_error("B2 retry loop exhausted");
    }
};

class RemoteGGUF {
    B2S3 b2_; std::string file_name_; uint64_t base_=0,size_=0,generation_=0;
    std::unordered_map<uint64_t,std::string> pages_;
    size_t cache_limit_; mutable std::mutex mutex_;
    struct CacheItem { std::shared_ptr<std::vector<uint8_t>> data; std::list<uint64_t>::iterator it; };
    mutable std::list<uint64_t> lru_; mutable std::unordered_map<uint64_t,CacheItem> cache_;
    std::shared_ptr<std::vector<uint8_t>> fetch_page(uint64_t page) const {
        { std::lock_guard<std::mutex> lock(mutex_); auto it=cache_.find(page); if(it!=cache_.end()){lru_.splice(lru_.begin(),lru_,it->second.it);return it->second.data;} }
        auto p=pages_.find(page); std::vector<uint8_t> raw;
        if(p==pages_.end()) raw=kZeroPage;
        else {
            auto packed=b2_.get(env("RAM16_B2_PREFIX","ram16")+"/pages/"+p->second+".z");
            raw.resize(kPageSize); uLongf out=kPageSize;
            int z=uncompress(raw.data(),&out,packed.data(),packed.size());
            if(z!=Z_OK||out!=kPageSize) throw std::runtime_error("zlib page decompression failed at virtual page "+std::to_string(page));
            if(sha256(raw)!=p->second) throw std::runtime_error("SHA-256 mismatch for virtual page "+std::to_string(page));
        }
        auto data=std::make_shared<std::vector<uint8_t>>(std::move(raw));
        { std::lock_guard<std::mutex> lock(mutex_); auto existing=cache_.find(page); if(existing!=cache_.end()){lru_.splice(lru_.begin(),lru_,existing->second.it);return existing->second.data;}
          lru_.push_front(page); cache_[page]={data,lru_.begin()};
          while(cache_.size()>cache_limit_){auto old=lru_.back();lru_.pop_back();cache_.erase(old);} } return data;
    }
    static uint32_t u32le(const std::vector<uint8_t>&b,size_t o){if(o+4>b.size())throw std::runtime_error("short superblock");return (uint32_t)b[o]|((uint32_t)b[o+1]<<8)|((uint32_t)b[o+2]<<16)|((uint32_t)b[o+3]<<24);}
public:
    explicit RemoteGGUF(size_t cache_pages):cache_limit_(std::max<size_t>(2,cache_pages)) {
        const std::string prefix=env("RAM16_B2_PREFIX","ram16");
        auto cur_bytes=b2_.get(prefix+"/CURRENT"); std::string cur_text(cur_bytes.begin(),cur_bytes.end());
        std::istringstream cur_stream(cur_text); ptree ptr; boost::property_tree::read_json(cur_stream, ptr);
        const std::string manifest_key=ptr.get<std::string>("manifest");
        if(manifest_key.rfind(prefix+"/manifests/",0)!=0) throw std::runtime_error("CURRENT points outside the RAM16 manifest prefix");
        auto manifest_raw=b2_.get(manifest_key);
        if(sha256(manifest_raw)!=ptr.get<std::string>("sha256")) throw std::runtime_error("CURRENT manifest SHA-256 mismatch");
        std::string manifest_text(manifest_raw.begin(),manifest_raw.end()); std::istringstream manifest_stream(manifest_text); ptree manifest; boost::property_tree::read_json(manifest_stream,manifest);
        if(manifest.get<int>("format")!=2||manifest.get<uint64_t>("virtual_size")!=kVirtualSize||manifest.get<size_t>("page_size")!=kPageSize||manifest.get<uint64_t>("page_count")!=kVirtualSize/kPageSize) throw std::runtime_error("unsupported RAM16 B2 page geometry/manifest");
        generation_=manifest.get<uint64_t>("generation");
        if(generation_!=ptr.get<uint64_t>("generation")) throw std::runtime_error("CURRENT generation mismatch");
        for(const auto &item:manifest.get_child("pages")){uint64_t n=std::stoull(item.first);const std::string digest=item.second.get_value<std::string>();if(n>=kVirtualSize/kPageSize)throw std::runtime_error("page index out of range");if(digest.size()!=64||digest.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("invalid SHA-256 page reference");pages_[n]=digest;}
        auto sb=fetch_page(0); if(std::memcmp(sb->data(),"RAM16MAP",8)!=0) throw std::runtime_error("RAM16 superblock missing");
        uint32_t n=u32le(*sb,8); if(n==0||n>kPageSize-12)throw std::runtime_error("invalid RAM16 superblock length");
        std::string meta_text(sb->begin()+12,sb->begin()+12+n); std::istringstream meta_stream(meta_text); ptree meta; boost::property_tree::read_json(meta_stream,meta);
        const auto &weight_region=meta.get_child("layout.weights");
        if(weight_region.empty()) throw std::runtime_error("RAM16 weight region missing");
        const uint64_t weight_start_page=weight_region.begin()->second.get_value<uint64_t>();
        if(weight_start_page>=kVirtualSize/kPageSize) throw std::runtime_error("RAM16 weight base page is out of range");
        base_=weight_start_page*kPageSize;
        size_=meta.get<uint64_t>("model.size");
        file_name_=meta.get<std::string>("model.filename","model.gguf");
        if(size_<24||size_>kVirtualSize-base_) throw std::runtime_error("GGUF region exceeds RAM16 virtual address space");
        if(cache_limit_>128) throw std::runtime_error("native page cache is capped at 128 pages (512 MiB)");
        std::cerr<<"RAM16 native GGUF: gen="<<generation_<<" file="<<file_name_<<" size="<<size_<<" bytes, mapped pages="<<pages_.size()<<", page cache="<<cache_limit_<<"\n";
    }
    uint64_t size()const{return size_;}
    const std::string& name()const{return file_name_;}
    int read(char *out,size_t n,off_t off)const {
        if(off<0) return -EINVAL;
        if((uint64_t)off>=size_) return 0;
        uint64_t pos=(uint64_t)off, remain=std::min<uint64_t>(n,size_-pos); size_t copied=0;
        try {
            while(remain){uint64_t absolute=base_+pos,page=absolute/kPageSize;size_t po=(size_t)(absolute%kPageSize);size_t take=(size_t)std::min<uint64_t>(remain,kPageSize-po);
                auto data=fetch_page(page); std::memcpy(out+copied,data->data()+po,take);pos+=take;copied+=take;remain-=take;}
            return (int)copied;
        } catch(const std::exception&e){std::cerr<<"GGUF page read error: "<<e.what()<<"\n";return -EIO;}
    }
};

std::unique_ptr<RemoteGGUF> g_model;
std::string g_mount_file="model.gguf";
static int fs_getattr(const char *path,struct stat *st,struct fuse_file_info*) {
    std::memset(st,0,sizeof(*st)); st->st_uid=getuid();st->st_gid=getgid();st->st_blksize=4096;
    if(std::strcmp(path,"/")==0){st->st_mode=S_IFDIR|0555;st->st_nlink=2;return 0;}
    if(path[0]=='/'&&std::string(path+1)==g_mount_file){st->st_mode=S_IFREG|0444;st->st_nlink=1;st->st_size=(off_t)g_model->size();st->st_blocks=(st->st_size+511)/512;return 0;}
    return -ENOENT;
}
static int fs_readdir(const char *path,void *buf,fuse_fill_dir_t filler,off_t,struct fuse_file_info*,enum fuse_readdir_flags){
    if(std::strcmp(path,"/")!=0) return -ENOENT;
    filler(buf,".",nullptr,0,(fuse_fill_dir_flags)0);
    filler(buf,"..",nullptr,0,(fuse_fill_dir_flags)0);
    filler(buf,g_mount_file.c_str(),nullptr,0,(fuse_fill_dir_flags)0);
    return 0;
}
static int fs_open(const char *path,struct fuse_file_info *fi){if(path[0]!='/'||std::string(path+1)!=g_mount_file)return -ENOENT;if((fi->flags&O_ACCMODE)!=O_RDONLY)return -EROFS;return 0;}
static int fs_read(const char*,char *buf,size_t size,off_t off,struct fuse_file_info*){return g_model->read(buf,size,off);}
static void *fs_init(struct fuse_conn_info *conn,struct fuse_config *cfg){(void)conn;cfg->kernel_cache=1;cfg->use_ino=1;cfg->entry_timeout=60.0;cfg->attr_timeout=60.0;return nullptr;}
static void fs_destroy(void*){g_model.reset();}
void usage(const char *p){std::cerr<<"Usage: "<<p<<" MOUNTPOINT [--file NAME] [--cache-pages N] [--foreground]\nB2 credentials are read from B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET, B2_ENDPOINT.\n";}
}
int main(int argc,char **argv){
    std::string mountpoint; size_t cache_pages=8;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];if(a=="--file"&&i+1<argc)g_mount_file=argv[++i];
        else if(a=="--cache-pages"&&i+1<argc)cache_pages=std::stoul(argv[++i]);
        else if(a=="--foreground") {}
        else if(a=="--help"){usage(argv[0]);return 0;}
        else if(!a.empty()&&a[0]=='-'){usage(argv[0]);return 2;}
        else if(mountpoint.empty())mountpoint=a;else{usage(argv[0]);return 2;}
    }
    if(mountpoint.empty()||g_mount_file.empty()||g_mount_file.find('/')!=std::string::npos){usage(argv[0]);return 2;}
    try{g_model=std::make_unique<RemoteGGUF>(cache_pages);}catch(const std::exception&e){std::cerr<<"B2 mount initialization failed: "<<e.what()<<"\n";return 1;}
    struct fuse_operations ops{};ops.getattr=fs_getattr;ops.readdir=fs_readdir;ops.open=fs_open;ops.read=fs_read;ops.init=fs_init;ops.destroy=fs_destroy;
    std::vector<std::string> args={argv[0],"-f","-o","ro,default_permissions,attr_timeout=60,entry_timeout=60",mountpoint};
    std::vector<char*> cargs;for(auto &s:args)cargs.push_back(s.data());
    return fuse_main((int)cargs.size(),cargs.data(),&ops,nullptr);
}
