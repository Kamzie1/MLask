#include "ConfusionMatrix.hpp"
#include <string>

namespace mlask{
void ConfusionMatrix::evaluate(int prediction, int expected){
    if (prediction == expected){
        if(expected == 0) TN += 1;
        else TP +=1;

    }
    else{
        if(expected == 0) FP += 1;
        else FN +=1;
    }
}
float_t ConfusionMatrix::accuracy()const{
    if (TP + TN + FP + FN == 0) return 0;
    return (float_t)(TP+TN)/(TP + TN + FP + FN);
}
float_t ConfusionMatrix::precision()const{
    if (TP + FP == 0) return 0;
    return (float_t)TP/(TP + FP);
}
float_t ConfusionMatrix::recall()const{
    if (TP + FN == 0) return 0;
    return (float_t)TP/(TP + FN);
}
float_t ConfusionMatrix::F1_score()const{
    float_t precision_ = precision();
    float_t recall_ = recall();
    if (precision_ + recall_ == 0) return 0;
    return 2 * (float_t)precision_ * recall_/(precision_ + recall_);
}
std::ostream & operator<<(std::ostream &os, const ConfusionMatrix& conf){
    os << "Confiusion Matrix: \n";
    os << "--------------\n";
    os << "|TP: " <<std::to_string(conf.TP) << "|TN: " <<std::to_string(conf.TN)<< "|\n";
    os << "--------------\n";
    os << "|FP: " <<std::to_string(conf.FP) <<"|FN: " <<std::to_string(conf.FN) <<"|\n";
    os << "--------------\n";

    os<<"Accuracy: "<<std::to_string(conf.accuracy()) <<" Precision: " << std::to_string(conf.precision()) << " Recall: "<<std::to_string(conf.recall()) << " F1 score: " << std::to_string(conf.F1_score());
    return os;
}
}
