#include "Variable.hpp"
#include<iostream>

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
std::shared_ptr<Variable> Variable::operator+(std::shared_ptr<Variable> other) {
    auto res = std::make_shared<Variable>(this->value + other->value);
    res->addDependency(shared_from_this(), 1.0);
    res->addDependency(other, 1.0);
    return res;
}


std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
    auto res = std::make_shared<Variable>(var1->getValue() + var2->getValue());
    res->addDependency(var1, 1.0);
    res->addDependency(var2, 1.0);
    return res;
}

// Difference operator overloadings
std::shared_ptr<Variable> Variable::operator-(std::shared_ptr<Variable> other) {
    auto res = std::make_shared<Variable>(this->value - other->value);
    res->addDependency(shared_from_this(), 1.0);
    res->addDependency(other, -1.0);
    return res;
}


std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
    auto res = std::make_shared<Variable>(var1->getValue() - var2->getValue());
    res->addDependency(var1, 1.0);
    res->addDependency(var2, -1.0);
    return res;
}

// Multiply operator overloadings
std::shared_ptr<Variable> Variable::operator*(std::shared_ptr<Variable> other) {
    auto res = std::make_shared<Variable>(this->value * other->value);
    res->addDependency(shared_from_this(), other->value);
    res->addDependency(other, this->value);
    return res;
}


std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
    auto res = std::make_shared<Variable>(var1->getValue() * var2->getValue());
    res->addDependency(var1, var2->getValue());
    res->addDependency(var2, var1->getValue());
    return res;
}

// Divide operator overloadings
std::shared_ptr<Variable> Variable::operator/(std::shared_ptr<Variable> other) {
    if (other->value == 0.0) {
        throw std::runtime_error("Division by zero!");
    }
    auto res = std::make_shared<Variable>(this->value / other->value);
    res->addDependency(shared_from_this(), 1.0 / other->value);
    res->addDependency(other, -this->value / (pow(other->value, 2.0)));
    return res;
}


std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2) {
    if (var2->getValue() == 0.0) {
        throw std::runtime_error("Division by zero!");
    }
    auto res = std::make_shared<Variable>(var1->getValue() / var2->getValue());
    res->addDependency(var1, 1.0 / var2->getValue());
    res->addDependency(var2, -var1->getValue() / (pow(var2->getValue(), 2.0)));
    return res;
}

void Variable::backward(double adjoint) {
    this->grad += adjoint;
    std::cout << "Computing gradient for node of value " << this->value << " and grad " << this->grad << "(" << this << ")" << std::endl;
    for (auto& dep : this->deps) {
        auto currentDep = dep.dependentVariable;
        std::cout << "Going to dep with value " << currentDep->getValue() << "(" << &currentDep << ")" << std::endl;
        currentDep->backward(dep.adjoint * adjoint);
    }
}

std::ostream& operator<<(std::ostream& os, std::shared_ptr<Variable> var){
    os << "Variable(value=" << var->getValue() << ", grad=" << var->getGrad() << ")";
    return os;
}
