#include"Variable.hpp"
#include <iostream>

int main(){
    auto x = std::make_shared<Variable>(1.0);
    auto y = std::make_shared<Variable>(2.0);
    auto z = std::make_shared<Variable>(0.25);
    auto k = ((x * y) + x)*z/y;
    k->backward();
    std::cout << z << std::endl;
    std::cout << x << std::endl;
    std::cout << y << std::endl;
}


