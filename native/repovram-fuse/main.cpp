// Read-only FUSE file over repovram release bins, for llama.cpp mmap.
// Native reader: libcurl range pages + kernel page cache + hot-page pinning.
#define FUSE_USE_VERSION 35
#include <fuse3/fuse.h>
#include <curl/curl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr size_t PAGE = 4u << 20;
constexpr size_t BIN_HDR = 4u << 20;
constexpr size_t DEFAULT_PPS = 510;

std::string envs(const char *k, const std::string &d = {}) {
    const char *v = std::getenv(k);
    return v && *v ? std::string(v) : d;
}

size_t write_cb(char *p, size_t a, size_t b, void *u) {
    auto *v = static_cast<std::vector<uint8_t> *>(u);
    v->insert(v->end(), p, p + a * b);
    return a * b;
}

class Http {
    std::string repo_, tag_, token_;
public:
    Http(std::string repo, std::string tag, std::string token)
        : repo_(std::move(repo)), tag_(std::move(tag)), token_(std::move(token)) {
        static std::once_flag once;
        std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    }

    // Returns empty vector for a missing bin/page (404/416).
    std::vector<uint8_t> range(size_t bin, uint64_t off, size_t n) const {
        char name[32];
        std::snprintf(name, sizeof(name), "mem-%02zu.bin", bin);
        const std::string url = "https://github.com/" + repo_ + "/releases/download/" +
                                tag_ + "/" + name;
        for (int attempt = 0; attempt < 5; ++attempt) {
            CURL *c = curl_easy_init();
            if (!c) throw std::runtime_error("curl_easy_init failed");
            std::vector<uint8_t> out;
            std::string range = "bytes=" + std::to_string(off) + "-" +
                                std::to_string(off + n - 1);
            struct curl_slist *headers = nullptr;
            if (!token_.empty())
                headers = curl_slist_append(headers, ("Authorization: Bearer " + token_).c_str());
            curl_easy_setopt(c, CURLOPT_URL, url.c_str());
            curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(c, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
            curl_easy_setopt(c, CURLOPT_MAXREDIRS, 5L);
            curl_easy_setopt(c, CURLOPT_RANGE, range.c_str());
            curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
            curl_easy_setopt(c, CURLOPT_WRITEDATA, &out);
            curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
            curl_easy_setopt(c, CURLOPT_TIMEOUT, 180L);
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(c, CURLOPT_USERAGENT, "repovram-fuse/1.0");
            CURLcode rc = curl_easy_perform(c);
            long status = 0;
            curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
            curl_slist_free_all(headers);
            curl_easy_cleanup(c);
            if (rc == CURLE_OK && (status == 404 || status == 416)) return {};
            if (rc == CURLE_OK && (status == 200 || status == 206)) {
                if (status == 200 && off) {
                    if (out.size() <= off) return {};
                    out.erase(out.begin(), out.begin() + off);
                }
                if (out.size() < n) out.resize(n, 0);
                else if (out.size() > n) out.resize(n);
                return out;
            }
            const bool retry = rc != CURLE_OK || status == 429 || status == 408 ||
                               (status >= 500 && status < 600);
            if (!retry || attempt == 4) {
                std::ostringstream e;
                e << "HTTP " << status << " " << curl_easy_strerror(rc) << " " << name
                  << " off=" << off << " url=" << url;
                throw std::runtime_error(e.str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(300 * (1 << attempt)));
        }
        throw std::runtime_error("retry exhausted");
    }
};

class Model {
    Http http_;
    uint64_t base_page_, size_;
    size_t pps_, ns_, cache_limit_, pin_limit_;
    mutable std::mutex mu_;
    mutable std::list<uint64_t> lru_;
    struct Entry {
        std::shared_ptr<std::vector<uint8_t>> data;
        uint32_t hits = 0;
        bool pinned = false;
        std::list<uint64_t>::iterator it;
    };
    mutable std::unordered_map<uint64_t, Entry> cache_;
    mutable uint64_t reads_ = 0, bytes_ = 0, faults_ = 0;

    std::shared_ptr<std::vector<uint8_t>> page(uint64_t p) const {
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = cache_.find(p);
            if (it != cache_.end()) {
                it->second.hits++;
                lru_.splice(lru_.begin(), lru_, it->second.it);
                if (!it->second.pinned && pin_limit_ && it->second.hits >= 2) {
                    size_t pinned = 0;
                    for (auto &kv : cache_) pinned += kv.second.pinned;
                    if (pinned < pin_limit_) it->second.pinned = true;
                }
                return it->second.data;
            }
        }
        const size_t bin = p / pps_;
        std::vector<uint8_t> raw;
        if (bin < ns_) {
            const uint64_t off = BIN_HDR + (p % pps_) * PAGE;
            try {
                raw = http_.range(bin, off, PAGE);
            } catch (const std::exception &e) {
                std::cerr << "page " << p << ": " << e.what() << "\n";
            }
        }
        if (raw.size() != PAGE) raw.assign(PAGE, 0);
        auto data = std::make_shared<std::vector<uint8_t>>(std::move(raw));
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = cache_.find(p);
            if (it != cache_.end()) return it->second.data;
            lru_.push_front(p);
            cache_[p] = Entry{data, 1, false, lru_.begin()};
            faults_++;
            while (cache_.size() > cache_limit_) {
                bool removed = false;
                for (auto rit = lru_.rbegin(); rit != lru_.rend(); ++rit) {
                    auto ci = cache_.find(*rit);
                    if (ci == cache_.end()) continue;
                    if (ci->second.pinned) continue;
                    cache_.erase(ci);
                    lru_.erase(std::next(rit).base());
                    removed = true;
                    break;
                }
                if (!removed) break;
            }
        }
        return data;
    }

public:
    Model(std::string repo, std::string tag, std::string token, uint64_t base_page,
          uint64_t size, size_t pps, size_t ns, size_t cache_limit, size_t pin_limit)
        : http_(std::move(repo), std::move(tag), std::move(token)), base_page_(base_page),
          size_(size), pps_(pps), ns_(ns), cache_limit_(std::max<size_t>(32, cache_limit)),
          pin_limit_(std::min(pin_limit, cache_limit_)) {
        std::cerr << "repovram-fuse: size=" << size_ << " base_page=" << base_page_
                  << " pps=" << pps_ << " ns=" << ns_ << " cache=" << cache_limit_
                  << " pin=" << pin_limit_ << "\n";
    }

    uint64_t size() const { return size_; }

    int read(char *out, size_t n, off_t off) const {
        if (off < 0) return -EINVAL;
        if ((uint64_t)off >= size_) return 0;
        uint64_t pos = (uint64_t)off, remain = std::min<uint64_t>(n, size_ - pos);
        size_t copied = 0;
        while (remain) {
            const uint64_t abs = base_page_ * PAGE + pos;
            const uint64_t p = abs / PAGE;
            const size_t po = (size_t)(abs % PAGE);
            const size_t take = (size_t)std::min<uint64_t>(remain, PAGE - po);
            auto d = page(p);
            std::memcpy(out + copied, d->data() + po, take);
            pos += take;
            copied += take;
            remain -= take;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            reads_++;
            bytes_ += copied;
            if (reads_ % 256 == 0)
                std::cerr << "fuse: reads=" << reads_ << " MiB=" << bytes_ / (1u << 20)
                          << " faults=" << faults_ << " cached=" << cache_.size()
                          << " pinned=" << std::count_if(cache_.begin(), cache_.end(),
                              [](const auto &kv) { return kv.second.pinned; }) << "\n";
        }
        return (int)copied;
    }

    void heat(const std::string &path) const {
        std::ofstream f(path);
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto &kv : cache_) f << kv.first << "\t" << kv.second.hits << "\n";
        f.close();
        std::cerr << "heatmap " << path << " pages=" << cache_.size() << "\n";
    }
};

std::unique_ptr<Model> g_model;
std::string g_name = "model.gguf";

int fs_getattr(const char *path, struct stat *st, struct fuse_file_info *) {
    std::memset(st, 0, sizeof(*st));
    st->st_uid = getuid(); st->st_gid = getgid();
    if (std::strcmp(path, "/") == 0) { st->st_mode = S_IFDIR | 0555; st->st_nlink = 2; return 0; }
    if (std::string(path) == "/" + g_name) {
        st->st_mode = S_IFREG | 0444; st->st_nlink = 1;
        st->st_size = (off_t)g_model->size(); st->st_blksize = PAGE;
        st->st_blocks = (st->st_size + 511) / 512; return 0;
    }
    return -ENOENT;
}
int fs_readdir(const char *, void *buf, fuse_fill_dir_t filler, off_t,
               struct fuse_file_info *, enum fuse_readdir_flags) {
    filler(buf, ".", nullptr, 0, (fuse_fill_dir_flags)0);
    filler(buf, "..", nullptr, 0, (fuse_fill_dir_flags)0);
    filler(buf, g_name.c_str(), nullptr, 0, (fuse_fill_dir_flags)0);
    return 0;
}
int fs_open(const char *path, struct fuse_file_info *fi) {
    if (std::string(path) != "/" + g_name) return -ENOENT;
    if ((fi->flags & O_ACCMODE) != O_RDONLY) return -EROFS;
    return 0;
}
int fs_read(const char *, char *buf, size_t size, off_t off, struct fuse_file_info *) {
    return g_model->read(buf, size, off);
}
void *fs_init(struct fuse_conn_info *, struct fuse_config *cfg) {
    cfg->kernel_cache = 1;
    cfg->auto_cache = 1;
    cfg->entry_timeout = 60.0;
    cfg->attr_timeout = 60.0;
    cfg->use_ino = 1;
    return nullptr;
}
void fs_destroy(void *) {
    if (g_model) {
        try { g_model->heat(envs("REPOVRAM_FUSE_HEAT", "/tmp/repovram-fuse-heat.tsv")); }
        catch (...) {}
        g_model.reset();
    }
}
void usage(const char *p) {
    std::cerr << "Usage: " << p << " MOUNTPOINT --repo OWNER/REPO --tag TAG "
                 "--base PAGE --size BYTES [--pps 510] [--ns 16] "
                 "[--cache-pages 512] [--pin-pages 256] [--file NAME]\n";
}
} // namespace

int main(int argc, char **argv) {
    std::string mount, repo, tag, file = "model.gguf";
    uint64_t base = 1, size = 0, pps = DEFAULT_PPS, ns = 16, cache = 512, pin = 256;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value"); return argv[++i]; };
        try {
            if (a == "--repo") repo = val();
            else if (a == "--tag") tag = val();
            else if (a == "--base") base = std::stoull(val());
            else if (a == "--size") size = std::stoull(val());
            else if (a == "--pps") pps = std::stoull(val());
            else if (a == "--ns") ns = std::stoull(val());
            else if (a == "--cache-pages") cache = std::stoull(val());
            else if (a == "--pin-pages") pin = std::stoull(val());
            else if (a == "--file") file = val();
            else if (a == "--foreground") {}
            else if (a == "--help") { usage(argv[0]); return 0; }
            else if (!a.empty() && a[0] == '-') { usage(argv[0]); return 2; }
            else if (mount.empty()) mount = a;
            else { usage(argv[0]); return 2; }
        } catch (const std::exception &e) { std::cerr << e.what() << "\n"; return 2; }
    }
    if (mount.empty() || repo.empty() || tag.empty() || !size || file.find('/') != std::string::npos) {
        usage(argv[0]); return 2;
    }
    g_name = file;
    try {
        g_model = std::make_unique<Model>(repo, tag, envs("GH_TOKEN"), base, size, pps, ns, cache, pin);
    } catch (const std::exception &e) {
        std::cerr << "mount init failed: " << e.what() << "\n";
        return 1;
    }
    struct fuse_operations ops{};
    ops.getattr = fs_getattr; ops.readdir = fs_readdir; ops.open = fs_open;
    ops.read = fs_read; ops.init = fs_init; ops.destroy = fs_destroy;
    std::vector<std::string> args = {argv[0], "-f", "-o", "ro,default_permissions", mount};
    std::vector<char *> cargs;
    for (auto &s : args) cargs.push_back(s.data());
    return fuse_main((int)cargs.size(), cargs.data(), &ops, nullptr);
}
