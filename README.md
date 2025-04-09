# 🧠 adbind

![adbind mascot](adbindini_brainini.png)

**adbind** is a reverse-mode automatic differentiation library written in both **C++** and **Python**, featuring seamless **Python bindings** via `pybind11`.

---

## ✨ Features

- Reverse-mode autodiff engine
- C++ backend for performance
- Python API for usability
- Lightweight and easy to extend
- Python bindings powered by `pybind11`

---

## 🛠 Installation

### Only CPP code
```bash
cd src/cpp
make        # Build the C++ library
make clean  # (Optional) Clean build files
```

### Python binding

use uv:
```bash
uv venv
source .venv/bin/activate
```

Then within `src/cpp` run:
```bash
python setup_adbind.py build_ext
python setup_adbind.py install
```


