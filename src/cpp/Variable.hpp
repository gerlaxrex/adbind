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
    // Operators and functions
    std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> other);
    std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> other);
    std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> other);
    std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> other);
    // with only doubles to emulate python 
    // Variable operator+(const double other) const;
    // Variable operator*(const double other) const;
    // Variable operator-(const double other) const;
    // Variable operator/(const double other) const;
    void backward(double adjoint=1.0);
};

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);
std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> var1, std::shared_ptr<Variable> var2);

std::ostream& operator<<(std::ostream& os, std::shared_ptr<Variable> var);
// Variable log(const Variable& other);
// Variable pow(const Variable& other);
// Variable sin(const Variable& other);
// Variable cos(const Variable& other);
// Variable relu(const Variable& other);
// Variable log(const double other) ;
// Variable pow(const double other) ;
// Variable sin(const double other) ;
// Variable cos(const double other) ;
// Variable relu(const double other) ;



#endif