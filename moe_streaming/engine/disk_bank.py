"""
DiskExpertBank — the heart of the engine.

Every expert's weights live in ONE flat file on disk (experts.bin). They are NOT
kept in RAM. When the router picks expert #i, *we* read only that expert's slice
off disk, hand it to the matmul, and (optionally) drop it again so the next token
must re-read it from disk.

Two real, measurable effects:
  • we only ever touch the bytes of the experts the router selected (top-k), and
  • with streaming ON we force genuine DISK reads — major page faults you can watch
    climb in /proc/self/stat — instead of silently living in the OS page cache.

Mechanism:
  - the file is memory-mapped (MAP_SHARED, read-only); reading a slice = a page
    access, which page-faults the bytes in (MAJOR fault if not cached = from disk).
  - posix_fadvise(DONTNEED) evicts a byte-range from the OS page cache, so the very
    next access has to come from the physical disk again. This is how we make
    "streaming from disk" real without needing root to drop caches.
"""
from __future__ import annotations

import mmap
import os
import numpy as np
import torch


def self_majflt() -> int:
    """This process's cumulative MAJOR page faults (pages read from disk)."""
    try:
        c = open("/proc/self/stat").read()
        return int(c[c.rfind(")") + 2:].split()[9])   # field 12 (majflt)
    except Exception:
        return 0


class DiskExpertBank:
    def __init__(self, path: str, dtype=np.float32):
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        self.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        # one read-only shared mapping of the whole file; slices fault in on access
        self.mm = mmap.mmap(self.fd, size, prot=mmap.PROT_READ)
        # entries[name] = dict(offset=bytes, num_experts, out, in_) — filled by register()
        self.entries: dict[str, dict] = {}
        self.stream = True          # True: evict after each read -> force disk re-reads
        self.bytes_read = 0         # bytes streamed for the current token (reset per token)
        self.majflt_start = self_majflt()   # major-fault baseline, to report a per-token delta
        self.last_experts: list[int] = []   # experts fired on the most recent token (for UI)

    def register(self, name: str, offset: int, num_experts: int, out: int, in_: int):
        self.entries[name] = {"offset": offset, "num_experts": num_experts,
                              "out": out, "in_": in_}

    def _evict(self, byte_off: int, nbytes: int):
        """Force a byte-range back onto disk. Order matters: MADV_DONTNEED first
        unmaps the pages from OUR mapping (fadvise won't evict still-mapped pages),
        then FADV_DONTNEED drops them from the OS page cache. Next access = a real
        disk read = a MAJOR page fault."""
        page = mmap.PAGESIZE
        a_off = byte_off & ~(page - 1)                       # align down to a page
        a_len = ((byte_off + nbytes + page - 1) & ~(page - 1)) - a_off
        try:
            self.mm.madvise(mmap.MADV_DONTNEED, a_off, a_len)
            os.posix_fadvise(self.fd, a_off, a_len, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass

    def cold(self):
        """Evict the WHOLE file so the first read of every expert is a disk read."""
        try:
            self.mm.madvise(mmap.MADV_DONTNEED)
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
        self.majflt_start = self_majflt()

    def read_expert(self, name: str, i: int) -> torch.Tensor:
        """Read expert #i of module `name` from disk -> [out, in] float32 tensor."""
        e = self.entries[name]
        n = e["out"] * e["in_"]
        nbytes = n * self.itemsize
        byte_off = e["offset"] + i * nbytes
        # np.frombuffer over the mmap: .copy() touches every page -> faults them in
        # (MAJOR faults, i.e. real disk reads, if we evicted them last time)
        arr = np.frombuffer(self.mm, dtype=self.dtype, count=n,
                            offset=byte_off).reshape(e["out"], e["in_"]).copy()
        self.bytes_read += nbytes
        if self.stream:
            self._evict(byte_off, nbytes)   # drop it so next token re-reads from disk
        return torch.from_numpy(arr)

    # ---- stats ----
    def reset_counters(self):
        """Call at the start of each token so bytes_read / majflt report that token."""
        self.bytes_read = 0
        self.majflt_start = self_majflt()

    def stats(self) -> dict:
        return {
            "bytes_read": self.bytes_read,                 # bytes streamed for this token
            "majflt": self_majflt() - self.majflt_start,   # disk page-ins WE caused this token
            "streaming": self.stream,
            "last_experts": self.last_experts,             # experts the router just used
            "total_experts_bytes": sum(                    # size of the whole expert bank
                e["num_experts"] * e["out"] * e["in_"] * self.itemsize
                for e in self.entries.values()),
        }
