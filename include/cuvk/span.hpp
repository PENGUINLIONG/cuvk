#pragma once
#include "cuvk/comdef.hpp"

L_CUVK_BEGIN_

template<typename T>
struct Span {
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using iterator = const value_type*;
  using const_iterator = const value_type*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using index_type = size_t;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  Span() noexcept : _base(nullptr), _size(0) {}
  Span(T* data, size_t size) noexcept : _base(data), _size(size) {}
  Span(const Span& b) : _base(b._base), _size(b._size) {}
  Span(Span&& b) :
    _base(std::exchange(b._base, nullptr)), _size(std::exchange(b._size, 0)) {}

  Span& operator=(const Span& b) noexcept {
    _base = b._base;
    _size = b._size;
    return *this;
  }
  Span& operator=(Span&& b) noexcept {
    _base = std::exchange(b._base, nullptr);
    _size = std::exchange(b._size, 0);
    return *this;
  }
  
  template<size_t TSize>
  Span(const std::array<T, TSize>& con, size_t n) noexcept :
    _base(con.data()),
    _size(n) {}
  template<size_t TSize>
  Span(const std::array<T, TSize>& con) noexcept :
    _base(con.data()),
    _size(con.size()) {}
  Span(const std::vector<T>& con) noexcept :
    _base(con.data()),
    _size(con.size()) {}

  size_t size() const noexcept { return _size; }
  size_t max_size() const noexcept { return _size; }
  const T* data() const noexcept { return _base; }

  const_iterator begin() const noexcept { return _base; }
  iterator begin() noexcept { return _base; }
  const_iterator cbegin() const noexcept { return _base; }

  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(cend());
  }
  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(cend());
  }

  const_iterator end() const noexcept { return _base + _size; }
  iterator end() noexcept { return _base + _size; }
  const_iterator cend() const noexcept { return _base + _size; }

  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(cbegin());
  }
  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(cbegin());
  }

  const T& operator[](size_t pos) const noexcept {
    if (pos >= _size) {
      LOG.error("index out of range");
      std::terminate();
    }
    return _base[pos];
  }

  bool operator==(const Span<T>& b) const noexcept {
    return std::equal(begin(), end(), b.begin(), b.end());
  }
  bool operator!=(const Span<T>& b) const noexcept { return !(*this == b); }

  void swap(Span<T>& b) noexcept {
    using std::swap;
    swap(_base, b._base);
    swap(_size, b._size);
  }

  bool empty() const noexcept {
    return _size != 0;
  }

private:
  const T* _base;
  size_t _size;
};

L_CUVK_END_
