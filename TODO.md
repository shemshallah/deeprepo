# Outstanding engineering work

- Validate the native FUSE3 mmap adapter on an actual GitHub-hosted runner and verify `/dev/fuse`, kernel page-fault behavior, and end-to-end llama.cpp inference against a real B2 bucket/model.
- Add B2-side quota accounting across all bucket prefixes and historical object versions; current page cap is scoped to visible `ram16/pages/` objects.
- Add authenticated/private model-source support (current `format-url` accepts public HTTP(S) URLs with Content-Length and byte-range resume).
- Benchmark page-fault latency, object GET count, kernel page-cache behavior, and cache sizing for sequential model load and random tensor access; tune bounded parallel read-ahead from measured results.
- Add end-to-end tests against a disposable B2 bucket and small real GGUF, including versioned-delete behavior, quota-boundary cases, runner interruption, and issue-chat inference.
- Consider a stronger externally enforced writer authority if the B2 key is shared with clients that may bypass the canonical GitHub workflow; current coordinator enforcement is application-level and cooperative.

- Verify SigV4 against the real Backblaze B2 endpoint and validate the optional least-privilege read-only key in a live runner.
