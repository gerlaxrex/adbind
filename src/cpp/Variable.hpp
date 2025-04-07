#ifndef VARIABLE_H
#define VARIABLE_H

#include <memory>
#include <vector>
#include <cmath>

class Variable : std::enable_shared_from_this<Variable> {
private:
    struct Dependency {
        std::shared_ptr<Variable> dependentVariable;
        float adjoint;
        Dependency(std::shared_ptr<Variable> var, double adj) : dependentVariable(var), adjoint(adj) {}
    };
    double value;
    double grad;
    std::vector<Dependency> deps;
    
public:
    void addDependency(std::shared_ptr<Variable> var, double adjoint);
    // Constructor
    Variable(double val);
    // setter/getters
    void setValue(double val);
    double getValue() const;
    double getGrad() const;
    void reset();
    // Decontsructor
    ~Variable() = default;
    void backward(double adjoint=1.0);
};

// Operators for sum, sub, mult, div also with scalars
std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var1);
std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);

std::shared_ptr<Variable> operator+(double var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator-(double var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator*(double var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator/(double var1, std::shared_ptr<Variable> var2);

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> var1, double var2);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var1, double var2);
std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> var1, double var2);
std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> var1, double var2);

std::ostream& operator<<(std::ostream& os, std::shared_ptr<Variable> var);
// Custom functions
std::shared_ptr<Variable> log(std::shared_ptr<Variable> input);
std::shared_ptr<Variable> pow(std::shared_ptr<Variable> base, std::shared_ptr<Variable> exponent);
std::shared_ptr<Variable> exp(std::shared_ptr<Variable> input);
std::shared_ptr<Variable> sin(std::shared_ptr<Variable> input);
std::shared_ptr<Variable> cos(std::shared_ptr<Variable> input);
std::shared_ptr<Variable> relu(std::shared_ptr<Variable> input);

#endif