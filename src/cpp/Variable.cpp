#include "Variable.hpp"
#include<iostream>
#include<cmath>

namespace adbind {

    Variable::Variable(double value) : value(value) {}

    double Variable::getGrad() const {
        return this->grad;
    }

    double Variable::getValue() const {
        return this->value;
    }

    void Variable::setValue(double value) {
        this->value = value;
    }

    void Variable::reset() {
        this->grad = 0.0;
    }

    void Variable::addDependency(std::shared_ptr<Variable> var, double adjoint) {
        deps.emplace_back(var, adjoint);
    }

    // Sum operator overloadings
    std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
        auto res = std::make_shared<Variable>(var1->getValue() + var2->getValue());
        res->addDependency(var1, 1.0);
        res->addDependency(var2, 1.0);
        return res;
    }

    std::shared_ptr<Variable> operator+(double scalar, std::shared_ptr<Variable> var) {
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator+(scalar_var, var);
    }

    std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> var, double scalar) {
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator+(var, scalar_var);
    }

    // Difference operator overloadings
    std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
        auto res = std::make_shared<Variable>(var1->getValue() - var2->getValue());
        res->addDependency(var1, 1.0);
        res->addDependency(var2, -1.0);
        return res;
    }

    std::shared_ptr<Variable> operator-(double scalar, std::shared_ptr<Variable> var) {
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator-(scalar_var, var);
    }

    std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var, double scalar) {
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator-(var, scalar_var);
    }

    // Single minus operator (negative unary)
    std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var1){
        auto negativeVar = std::make_shared<Variable>(var1->getValue());
        negativeVar->addDependency(var1, -1.0);
        return negativeVar;
    }

    // Multiply operator overloadings
    std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
        auto res = std::make_shared<Variable>(var1->getValue() * var2->getValue());
        res->addDependency(var1, var2->getValue());
        res->addDependency(var2, var1->getValue());
        return res;
    }

    std::shared_ptr<Variable> operator*(double scalar, std::shared_ptr<Variable> var) {
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator*(scalar_var, var);
    }

    std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> var, double scalar) {
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator*(var, scalar_var);
    }

    // Divide operator overloadings
    std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
        if (var2->getValue() == 0.0) {
            throw std::runtime_error("Division by zero!");
        }
        auto res = std::make_shared<Variable>(var1->getValue() / var2->getValue());
        res->addDependency(var1, 1.0 / var2->getValue());
        res->addDependency(var2, -var1->getValue() / (std::pow(var2->getValue(), 2.0)));
        return res;
    }

    std::shared_ptr<Variable> operator/(double scalar, std::shared_ptr<Variable> var) {
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator/(scalar_var, var);
    }

    std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> var, double scalar) {
        if (scalar == 0.0) {
            throw std::runtime_error("Division by zero!");
        }
        auto scalar_var = std::make_shared<Variable>(scalar);
        return operator/(var, scalar_var);
    }
    // Backward
    void Variable::backward(double adjoint) {
        this->grad += adjoint;
        // std::cout << "Computing gradient for node of value " << this->value << " and grad " << this->grad << "(" << this << ")" << std::endl;
        for (auto& dep : this->deps) {
            auto currentDep = dep.dependentVariable;
            // std::cout << "Going to dep with value " << currentDep->getValue() << "(" << &currentDep << ")" << std::endl;
            currentDep->backward(dep.adjoint * adjoint);
        }
    }

    std::ostream& operator<<(std::ostream& os, std::shared_ptr<Variable> var){
        os << "Variable(value=" << var->getValue() << ", grad=" << var->getGrad() << ")";
        return os;
    }

    std::shared_ptr<Variable> log(std::shared_ptr<Variable> input){
        if(input->getValue() <= 0){
            throw std::runtime_error("Logarithm of non-positive value is undefined");
        }
        auto logVar = std::make_shared<Variable>(std::log(input->getValue()));
        logVar->addDependency(input, 1.0 / input->getValue());
        return logVar;
    }


    std::shared_ptr<Variable> pow(std::shared_ptr<Variable> base, std::shared_ptr<Variable> exponent){
        auto expVar = std::make_shared<Variable>(std::pow(base->getValue(), exponent->getValue()));
        expVar->addDependency(base,  exponent->getValue() * std::pow(base->getValue(), exponent->getValue()-1));
        expVar->addDependency(exponent,  std::pow(base->getValue(), exponent->getValue()-1) * std::log(base->getValue()));
        return expVar;
    }

    std::shared_ptr<Variable> exp(std::shared_ptr<Variable> input){
        auto expVar = std::make_shared<Variable>(std::exp(input->getValue()));
        expVar->addDependency(input, expVar->getValue());
        return expVar;
    }

    std::shared_ptr<Variable> sin(std::shared_ptr<Variable> input){
        auto sinVar = std::make_shared<Variable>(std::sin(input->getValue()));
        sinVar->addDependency(input, std::cos(input->getValue()));
        return sinVar;
    }

    std::shared_ptr<Variable> cos(std::shared_ptr<Variable> input){
        auto cosVar = std::make_shared<Variable>(std::cos(input->getValue()));
        cosVar->addDependency(input, -std::sin(input->getValue()));
        return cosVar;
    }

    std::shared_ptr<Variable> relu(std::shared_ptr<Variable> input){
        auto reluVar = std::make_shared<Variable>(std::max(0.0, input->getValue()));
        reluVar->addDependency(input, input->getValue()>0? 1.0 : 0.0);
        return reluVar;
    }
}