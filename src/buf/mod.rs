//! 提供了和buffer交互的工具.
//!
//! buffer是一个包含了一系列字节的结构. 这些字节可能存储在连续的内存中, 也可能不连续.
//! 本模块提供了对buffer进行抽象,以及对buffer类型进行操作的traits和工具.
//!
//! # `Buf`, `BufMut`
//!
//! 这是两个重要的traits, 用于抽象地处理buffer. 它们可以被视为字节结构的迭代器.
//! 它们提供了在`Iterator`之上的性能, 因为提供了针对字节切片优化的API.
//!
//! See [`Buf`] and [`BufMut`] for more details.
//!
//! [rope]: https://en.wikipedia.org/wiki/Rope_(data_structure)

mod buf_impl;
mod buf_mut;
mod chain;
mod iter;
mod limit;
#[cfg(feature = "std")]
mod reader;
mod take;
mod uninit_slice;
mod vec_deque;
#[cfg(feature = "std")]
mod writer;

pub use self::buf_impl::Buf;
pub use self::buf_mut::BufMut;
pub use self::chain::Chain;
pub use self::iter::IntoIter;
pub use self::limit::Limit;
pub use self::take::Take;
pub use self::uninit_slice::UninitSlice;

#[cfg(feature = "std")]
pub use self::{reader::Reader, writer::Writer};
