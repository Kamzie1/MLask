#pragma once
#include <cstddef>
#include <iostream>

namespace mlask{
/** @brief Class used to write progress bars in the terminal */
class ProgressBar{
    inline static const std::size_t size = 100;
    inline static const std::string title = "PROGRESS BAR";
public:
    /** @brief Draws the progress bar
     * @param progress The progress value (0.0 to 1.0)
     */
    static void draw(float progress){
        std::cout<<"\033[2J\033[H"; /* clear screen */

        std::cout<<"\033[36m"; /* cyan color */
        for(std::size_t i=0;i<size-title.length();i++){
            std::cout<<'-';
            if (i == (size - title.length())/2){
                std::cout<<"\033[31m"<<"\033[5m"<<"\033[1m"<<title<<"\033[25m"<<"\033[22m";
                std::cout<<"\033[36m"; /* cyan color */
            }
        }
        std::cout<<std::endl;

        std::cout<<"\033[92m"; /* green color */
        progress = (std::size_t)(progress * size);
        std::size_t i;
        for(i=0;i<progress;i++){
            std::cout<<'|';
        }
        std::cout<<"\033[31m"; /* red color */
        for(;i<size;i++){
            std::cout<<'|';
        }
        std::cout<<std::endl;

        std::cout<<"\033[36m"; /* cyan color */
        for(std::size_t i=0;i<size;i++){
            std::cout<<'-';
        }
        std::cout<<std::endl;

        std::cout<<"\033[0m"; /* reset of styles */
    }
};
}
