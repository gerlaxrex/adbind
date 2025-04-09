#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <sstream>
#include "Variable.hpp"

namespace py = pybind11;

PYBIND11_MODULE(adbind, m) {
    m.doc() = "adbind: very simple reverse-mode autodiff with c++ bindings";
    
    py::class_<adbind::Variable, std::shared_ptr<adbind::Variable>>(m, "Variable")
        .def(py::init<double>())
        .def("get_grad", &adbind::Variable::getGrad)
        .def("get_value", &adbind::Variable::getValue)
        .def("set_value", &adbind::Variable::setValue)
        .def("reset", &adbind::Variable::reset)
        .def("backward", &adbind::Variable::backward, py::arg("adjoint") = 1.0)
        .def("__add__", [](const std::shared_ptr<adbind::Variable>& a, const std::shared_ptr<adbind::Variable>& b) { 
            return a + b; 
        }, py::is_operator())
        .def("__sub__", [](const std::shared_ptr<adbind::Variable>& a, const std::shared_ptr<adbind::Variable>& b) { 
            return a - b; 
        }, py::is_operator())
        .def("__mul__", [](const std::shared_ptr<adbind::Variable>& a, const std::shared_ptr<adbind::Variable>& b) { 
            return a * b; 
        }, py::is_operator())
        .def("__truediv__", [](const std::shared_ptr<adbind::Variable>& a, const std::shared_ptr<adbind::Variable>& b) { 
            return a / b; 
        }, py::is_operator())
        .def("__add__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return a + b; 
        }, py::is_operator())
        .def("__radd__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return b + a; 
        }, py::is_operator())
        .def("__sub__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return a - b; 
        }, py::is_operator())
        .def("__rsub__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return b - a; 
        }, py::is_operator())
        .def("__mul__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return a * b; 
        }, py::is_operator())
        .def("__rmul__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return b * a; 
        }, py::is_operator())
        .def("__truediv__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return a / b; 
        }, py::is_operator())
        .def("__rtruediv__", [](const std::shared_ptr<adbind::Variable>& a, double b) { 
            return b / a; 
        }, py::is_operator())
        // Unary minus
        .def("__neg__", [](const std::shared_ptr<adbind::Variable>& a) { 
            return -a; 
        }, py::is_operator())
        .def("__repr__", [](const std::shared_ptr<adbind::Variable>& v) {
            std::stringstream ss;
            ss << "Variable(value=" << v->getValue() << ", grad=" << v->getGrad() << ")";
            return ss.str();
        })
        .def("__pow__", [](const std::shared_ptr<adbind::Variable>& base, const std::shared_ptr<adbind::Variable>& exponent) {
            return adbind::pow(base, exponent);
        })
        .def("__pow__", [](const std::shared_ptr<adbind::Variable>& base, double exponent) {
            return adbind::pow(base, exponent);
        })
        ;
    
    m.def("log", &adbind::log);
    m.def("exp", &adbind::exp);
    m.def("sin", &adbind::sin);
    m.def("cos", &adbind::cos);
    m.def("relu", &adbind::relu);
}