#![allow(unknown_lints, unexpected_cfgs)]
#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms)]
#![doc(test(
    no_crate_inject,
    attr(deny(warnings, rust_2018_idioms), allow(dead_code, unused_variables))
))]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! 提供了对字节处理的抽象。
//!
//! `bytes` crate 提供了一个高效的字节缓冲区结构 [`Bytes`] 和用于处理缓冲区实现的 traits [`Buf`] 和 [`BufMut`].
//!
//! # `Bytes`
//!
//! `Bytes` 是一个高效的容器, 用于存储和操作连续内存片段(例如`slice`)。它主要用于网络编程，但也可能有其他用途。
//!
//! `Bytes` 值可以实现零拷贝网络编程，因为它允许多个 `Bytes` 对象指向同一块共享内存。
//!  这由引用计数跟踪并管理, 当该块内存不再需要时释放。
//!
//! 一个 `Bytes` 的handle可以直接从一个字节存储(例如 `&[u8]` 或 `Vec<u8>`)创建,
//! 但通常先创建一个 `BytesMut` 并写入数据。例如：
//!
//! ```rust
//! use bytes::{BytesMut, BufMut};
//!
//! let mut buf = BytesMut::with_capacity(1024);
//! buf.put(&b"hello world"[..]);
//! buf.put_u16(1234);
//!
//! let a = buf.split();
//! assert_eq!(a, b"hello world\x04\xD2"[..]);
//!
//! buf.put(&b"goodbye world"[..]);
//!
//! let b = buf.split();
//! assert_eq!(b, b"goodbye world"[..]);
//!
//! assert_eq!(buf.capacity(), 998);
//! ```
//! 在上面的例子中, 只有一个 1024 字节的缓冲区被分配。`a` 和 `b` 两个handle共享同一块内存，并维护了对该内存的视图。
//!
//! See the [struct docs](`Bytes`) for more details.
//!
//! # `Buf`, `BufMut`
//!
//! 这两个trait提供了对buffer的读写访问。底层存储可能是连续的，也可能不是。
//! 例如，`Bytes` 是一个保证连续内存的buffer,但是一个[rope]存储字节的实现可能是分散的。
//! `Buf` 和 `BufMut` 维护了一个游标，用于跟踪当前位置在底层字节存储中的位置。
//! 当字节被读取或写入时，游标会被移动。
//!
//! [rope]: https://en.wikipedia.org/wiki/Rope_(data_structure)
//!
//! ## Relation with `Read` and `Write`
//!
//! 乍看之下, `Buf` 和 `BufMut` 看起来很像 [`std::io::Read`] 和 [`std::io::Write`].
//! 但是, 它们有一些重要的区别:
//! 一个buffer是提供给`Read::read`和`Write::write`的参数的值.
//! `std::io::Read` 和 `std::io::Write` 可能会发起系统调用,并且可能会失败。
//! 而 `Buf` 和 `BufMut` 不会。

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod buf;
pub use crate::buf::{Buf, BufMut};

mod bytes;
mod bytes_mut;
mod fmt;
mod loom;
pub use crate::bytes::Bytes;
pub use crate::bytes_mut::BytesMut;

// Optional Serde support
#[cfg(feature = "serde")]
mod serde;

#[inline(never)]
#[cold]
fn abort() -> ! {
    #[cfg(feature = "std")]
    {
        std::process::abort();
    }

    #[cfg(not(feature = "std"))]
    {
        struct Abort;
        impl Drop for Abort {
            fn drop(&mut self) {
                panic!();
            }
        }
        let _a = Abort;
        panic!("abort");
    }
}

#[inline(always)]
#[cfg(feature = "std")]
fn saturating_sub_usize_u64(a: usize, b: u64) -> usize {
    use core::convert::TryFrom;
    match usize::try_from(b) {
        Ok(b) => a.saturating_sub(b),
        Err(_) => 0,
    }
}

#[inline(always)]
#[cfg(feature = "std")]
fn min_u64_usize(a: u64, b: usize) -> usize {
    use core::convert::TryFrom;
    match usize::try_from(a) {
        Ok(a) => usize::min(a, b),
        Err(_) => b,
    }
}

/// Panic with a nice error message.
#[cold]
fn panic_advance(idx: usize, len: usize) -> ! {
    panic!(
        "advance out of bounds: the len is {} but advancing by {}",
        len, idx
    );
}

#[cold]
fn panic_does_not_fit(size: usize, nbytes: usize) -> ! {
    panic!(
        "size too large: the integer type can fit {} bytes, but nbytes is {}",
        size, nbytes
    );
}

/// Precondition: dst >= original
///
/// The following line is equivalent to:
///
/// ```rust,ignore
/// self.ptr.as_ptr().offset_from(ptr) as usize;
/// ```
///
/// But due to min rust is 1.39 and it is only stabilized
/// in 1.47, we cannot use it.
#[inline]
fn offset_from(dst: *const u8, original: *const u8) -> usize {
    dst as usize - original as usize
}
